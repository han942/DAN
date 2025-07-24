import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # Default hparams
    parser.add_argument('--dataset', type=str,default='yelp2018', help='dataset name')
    parser.add_argument('--model', type=str, default='EASE')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--gpu', type=int, default=-1, help="Please give a value for gpu id")
    parser.add_argument('--testbatch', type=int, default=4096, help="the batch size of users for testing")
    parser.add_argument('--topks', nargs='?',default="[20, 100]", help="@k test list, e.g. [10,20,30,40,50]")
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--parallel', type=eval, default=False, choices=[True, False])
   
    parser.add_argument('--reg_p', type=float, default=100)
    parser.add_argument('--diag_const', type=eval, default=True, choices=[True, False])
    parser.add_argument('--drop_p', type=float, default=0)
    parser.add_argument('--xi', type=float, default=0.0)
    parser.add_argument('--relax_type', type=str,default='relative')
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--relax', type=eval, default=True, choices=[True, False])
    return parser.parse_args(args=[])

def get_default_config():
    """
    Jupyter Notebook을 위한 함수.
    모든 파라미터의 기본값을 딕셔너리로 반환합니다.
    """
    con = {
        'dataset': 'yelp2018',
        'model': 'EASE',
        'seed': 2020,
        'gpu': -1,
        'testbatch': 4096,
        'topks': "[20, 100]",
        'multicore': 0,
        'parallel': False,
        'reg_p': 100.0,
        'diag_const': True,
        'drop_p': 0.0,
        'xi': 0.0,
        'relax_type': 'relative',
        'alpha': 1.0,
        'beta': 0.5,
        'relax': True
    }
    return con