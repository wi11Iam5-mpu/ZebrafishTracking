from datetime import datetime
from functools import wraps, partial


def print_dict_kindly(input_data, is_sorted=False, is_show=True):
    """
    keyword should be a str type;
        this function will automatic determine that and convert
    noted:
        1. this function will change change the keyword order; updated: problem solved
        2. printing some built-in python objects is not supported
    :param input_data:
    :param is_sorted: whether or not to sort keywords
    :param is_show: whether or not to show details
    :return new_data: converted results
    """
    import json
    import copy

    if not isinstance(input_data, dict):
        raise TypeError("input should be dict type!")

    new_data = copy.deepcopy(input_data)

    def recursion_check(d):
        """
        Recursively check if the key value is a character,
        and replace it with a character if it is not
        """
        for k, v in d.items():
            if isinstance(v, dict):
                recursion_check(v)

            if not isinstance(k, str):
                d[str(k)] = d.pop(k)
            else:
                d[k] = d.pop(k)  # just for keeping the order
        return d

    new_data = recursion_check(new_data)
    if is_show:
        print(json.dumps(recursion_check(new_data), indent=4, sort_keys=is_sorted))
    return new_data


def check_keys(keys, param_dict):
    """
    Check if the required parameters exist in the configuration file
    """
    assert param_dict, "setting file is empty"
    s = set(keys)
    t = set(param_dict.keys())
    try:
        if s.issubset(t):
            if t - s:
                print(f"Warning: extra parameters [{t - s}]")
            return
        else:
            raise ValueError
    except ValueError:
        print(f"Error, {s - t} not in setting file,"
              f" besides keys is missing {t - s} ")


def time_count(func=None, is_print=True, unwrap=False):
    """
    Calculate the running time of the decorated codes
    """
    if func is None:
        return partial(time_count, is_print=is_print)

    if unwrap:
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        st = time.time()
        if is_print:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>> [{func.__name__}] time start: {datetime.now()}")
        func(*args, **kwargs)
        if is_print:
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>> [{func.__name__}] time cost: {(time.time() - st) : .2f}s')
        return

    return wrapper


def lazy_property(func):
    """
    This property is used to implement the lazy calculation
    """
    name = '_lazy_' + func.__name__

    @property
    def lazy(self):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            value = func(self)
            setattr(self, name, value)
            return value

    return lazy


if __name__ == '__main__':
    d1 = {
        1: None,
        "3": None,
        "4": None,
    }
    d2 = {
        "1": None,
        2: None,
        "3": d1,
    }
    print_dict_kindly(d2)
    check_keys(d1.keys(), d2)


    @time_count(is_print=True)
    def long_time():
        import time
        time.sleep(0.5)


    long_time()


    class Circle:
        @lazy_property
        def pi(self):
            print("&&&&")
            return 3 + 4


    a = Circle()
    print(a.pi)
    print(a.pi)
    print(a.pi)

    # print(inspect.signature(Stock))
    # s1 = Stock('ACME', 100, 490.1)
    # s2 = Stock('ACME', 100)
    # s3 = Stock('ACME', 100, 490.1, shares=50)
