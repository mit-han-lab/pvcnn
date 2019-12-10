import os

__all__ = ['set_cuda_visible_devices']


def set_cuda_visible_devices(devs):
    gpus = []
    for dev in devs.split(','):
        dev = dev.strip().lower()
        if dev == 'cpu':
            continue
        if dev.startswith('gpu'):
            dev = dev[3:]
        if '-' in dev:
            l, r = map(int, dev.split('-'))
            gpus.extend(range(l, r + 1))
        else:
            gpus.append(int(dev))

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
    return gpus
