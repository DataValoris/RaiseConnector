import functools
import sys
import time

import requests

from clients.utils import JSONSerializer as json


class RequestToApi:
    ATTEMPTS_COUNT = 5
    REPEATS_COUNT = 8640  # 1 day
    DEFAULT_TIMEOUT = 5 * 60  # 5 minutes
    REPEAT_TIME = 15  # 15 seconds
    SLEEP_TIME = 5  # 5 seconds
    REQUEST_TIMEOUT = 2  # 2 seconds

    def __init__(self, url, token):
        self.url = url
        self.token = token
        self.expired_token = False
        if self.check_connection():
            self.auth = self.get_token()

    class RepeatRequest(object):
        def __init__(self, attempts_count, sleep_time):
            self.attempts_count = attempts_count
            self.sleep_time = sleep_time

        def __call__(self, fn):
            @functools.wraps(fn)
            def decorated(*args, **kwargs):
                result = self._repeat(fn, *args, **kwargs)
                return result

            return decorated

        def _repeat(self, f, *args, **kwargs):
            attempt = 0
            while attempt < self.attempts_count:
                try:
                    result = f(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt < self.attempts_count:
                        time.sleep(self.sleep_time)
                        print(f"Error occures during {f.__name__}! Attempt {attempt} to run {f.__name__}")
                    else:
                        raise e
                else:
                    return result

    class ProcessAsyncRequest(object):
        def __init__(self, attempts_count, repeat_time):
            self.repeats_count = attempts_count
            self.repeat_time = repeat_time

        def __call__(self, fn):
            @functools.wraps(fn)
            def decorated(*args, **kwargs):
                result = self._repeat(fn, *args, **kwargs)
                return result

            return decorated

        def _repeat(self, f, *args, **kwargs):
            attempt = 0
            while attempt < self.repeats_count:
                try:
                    result = f(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt < self.repeats_count:
                        time.sleep(self.repeat_time)
                        print(f"Need more time for creating new generation. Attempt {attempt} to {f.__name__}")
                    else:
                        raise e
                else:
                    return result

    def check_response(self, response):
        if response.status_code == 401:
            if self.expired_token:
                raise Exception("Permission denied")
            else:
                if response.json().get("message") == "jwt expired":
                    print("Token expired")
                    self.auth = self.get_token()
                self.expired_token = True
        elif response.status_code in (402, 403, 404):
            print("############################### EXIT MESSAGE: ####################################")
            print(response.json().get("message"))
            sys.exit()
        elif response.status_code in (200, 201):
            self.expired_token = False
        else:
            raise Exception(response.text)

    repeat_request_if_failed = RepeatRequest(ATTEMPTS_COUNT, SLEEP_TIME)
    repeat_request_if_did_not_done = ProcessAsyncRequest(REPEATS_COUNT, REPEAT_TIME)

    def check_connection(self):
        attempt = 0
        while attempt < self.ATTEMPTS_COUNT:
            try:
                requests.get(self.url, timeout=self.REQUEST_TIMEOUT)
                print("Connection is established")
            except Exception as e:
                attempt += 1
                if attempt < self.ATTEMPTS_COUNT:
                    time.sleep(self.REQUEST_TIMEOUT)
                    print(f"Attempt {attempt} to establish a connection")
                else:
                    print(f"\nInternet connection error\n")
                    raise ConnectionError(e)
            else:
                return True

    @repeat_request_if_did_not_done
    def get_new_population(self, task_id, timeout, headers):
        response = requests.get(
            f"{self.url}engine/check/{task_id}/",
            timeout=timeout,
            headers=headers,
        )
        if response.status_code == 200 and response.json() == "FAILURE":
            response.status_code = 500
            response.reason = "Internal Server Error"
        elif response.status_code == 200 and response.json() == "PENDING" or response.status_code != 200:
            raise Exception("Result isn't ready")
        return response

    def init_project(self, source, project_id, initial_mutant):
        source = json.loads(str(source)) if isinstance(source, str) else source
        request_params = {"project_id": project_id, "source": source, "initial_mutant": initial_mutant}
        response = requests.post(
            f"{self.url}engine/genotype/",
            data=json.dumps(request_params),
            timeout=(self.DEFAULT_TIMEOUT, self.DEFAULT_TIMEOUT),
            headers={"Authorization": f"Bearer {self.auth}", "Content-type": "application/json"},
        )
        self.check_response(response)
        data = response.json()
        convert_coordinate_to_tuple(data["genotype"])  # transform list Coordinate to tuple
        return data

    @repeat_request_if_failed
    def start_project(self, project_id, instance):
        return self.modify_genotype(instance, project_id, True)

    @repeat_request_if_failed
    def modify_genotype(self, source=None, project_id=None, is_initial=False):
        source = json.loads(str(source)) if isinstance(source, str) else source
        request_params = {
            "project_id": project_id,
            "source": source,
            "is_initial": is_initial,
        }
        response = requests.post(
            f"{self.url}engine/async_population/",
            data=json.dumps(request_params),
            timeout=(self.DEFAULT_TIMEOUT, self.DEFAULT_TIMEOUT),
            headers={"Authorization": f"Bearer {self.auth}", "Content-type": "application/json"},
        )
        if response.status_code == 201:
            response = self.get_new_population(
                task_id=response.json().get("task_id"),
                timeout=(self.DEFAULT_TIMEOUT, self.DEFAULT_TIMEOUT),
                headers={"Authorization": f"Bearer {self.auth}"},
            )
        self.check_response(response)
        data = response.json()
        convert_coordinate_to_tuple(data)  # transform list Coordinate to tuple
        return data

    @repeat_request_if_failed
    def send_selection_cycle_result(self, project_id, request_params):
        response = requests.post(
            f"{self.url}engine/remote-logic/{project_id}/",
            data=json.dumps(request_params),
            headers={"Authorization": f"Bearer {self.auth}", "Content-type": "application/json"},
        )
        self.check_response(response)
        return response.json()

    @repeat_request_if_failed
    def get_project(self, project_id):
        response = requests.get(f"{self.url}projects/{project_id}/", headers={"Authorization": f"Bearer {self.auth}"})
        self.check_response(response)
        return response.json()

    @repeat_request_if_failed
    def modify_project(self, project_id, request_params):
        response = requests.patch(
            f"{self.url}projects/{project_id}/",
            data=json.dumps(request_params),
            headers={"Authorization": f"Bearer {self.auth}", "Content-type": "application/json"},
        )
        self.check_response(response)
        return response.json()

    @repeat_request_if_failed
    def create_mutant(self, request_params):
        response = requests.post(
            f"{self.url}projects-mutants/",
            data=json.dumps(request_params),
            headers={"Authorization": f"Bearer {self.auth}", "Content-type": "application/json"},
        )
        self.check_response(response)
        return response.json()

    @repeat_request_if_failed
    def get_mutant(self, mutant_id):
        response = requests.get(
            f"{self.url}projects-mutants/{mutant_id}/", headers={"Authorization": f"Bearer {self.auth}"}
        )
        self.check_response(response)
        return response.json()

    @repeat_request_if_failed
    def update_mutant(self, mutant_id, request_params):
        response = requests.patch(
            f"{self.url}projects-mutants/{mutant_id}/",
            data=json.dumps(request_params),
            headers={"Authorization": f"Bearer {self.auth}", "Content-type": "application/json"},
        )
        self.check_response(response)
        return response.json()

    @repeat_request_if_failed
    def send_cycle_info(self, request_params):
        response = requests.post(
            f"{self.url}projects-cycles/",
            data=json.dumps(request_params),
            headers={"Authorization": f"Bearer {self.auth}", "Content-type": "application/json"},
        )
        self.check_response(response)
        return response.json()

    @repeat_request_if_failed
    def check_client_version(self, version):
        response = requests.get(f"{self.url}connector-version/", headers={"Authorization": f"Bearer {self.auth}"})
        self.check_response(response)
        return response.json().get("current") == version

    @repeat_request_if_failed
    def get_token(self):
        request_params = {"strategy": "hash", "hash": self.token}
        response = requests.post(f"{self.url}authentication/", json=request_params)
        self.check_response(response)
        return response.json().get("accessToken")


def convert_coordinate_to_tuple(data):
    """
    transform list Coordinate to tuple, for example.
    some problem could be with keys = 'kernel_size', 'dilation_rate', 'strides', 'pool_size', 'Coordinate',
    'SelectionType', 'FromCoordinate' or 'From'
    :param data: dict contains key 'Coordinate', 'FromCoordinate' or 'From'
    >>> convert_coordinate_to_tuple({'Coordinate': [0.125, 0.5]})

    # will be {'Coordinate': (0.125, 0.5)}
    >>> convert_coordinate_to_tuple({'From': [[0.125, 0.5], [0.12, 0.75]]})

    # will be {'From': [(0.125, 0.5), (0.12, 0.75)]}
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ("Coordinate", "FromCoordinate"):
                # print("{0} : {1}".format(key, value))
                if isinstance(value, list) and len(value) == 2:
                    data[key] = tuple(value)
            elif key == "From":
                if isinstance(value, list):
                    data[key] = [tuple(item) for item in value]

            if isinstance(value, dict):
                convert_coordinate_to_tuple(value)
            elif isinstance(value, list):
                for item in value:
                    convert_coordinate_to_tuple(item)
    elif isinstance(data, list):
        for item in data:
            convert_coordinate_to_tuple(item)
