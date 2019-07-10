

class CameraParams:
    def __init__(self, k, std_read, std_adc, name=None):
        self.k = k
        self.std_read = std_read
        self.std_adc = std_adc
        self.name = name

def interpolate_camera(cam1: CameraParams, cam2: CameraParams, ratio=0.25):
    """
    cam1, cam2: two camera to interpolate with
    """
    calc = lambda x1, x2: x1 * ratio + x2 * (1-ratio)
    return CameraParams(
        k=[calc(k1, k2) for k1,k2 in zip(cam1.k, cam2.k)],
        std_adc=calc(cam1.std_adc, cam2.std_adc),
        std_read=calc(cam1.std_read, cam2.std_read),
        name=f'{cam1.name}-{ratio}-{cam2.name}')

Camera = {
    'SonyA7r1': CameraParams(
        k=[2.23647, 1.32064, 5.77612],
        std_read = 3.19798,
        std_adc = 6.60121,
        name='SonyA7r1'),
    'CanonT1': CameraParams(
        k=[4.33639, 2.56427, 9.48193],
        std_read = 0.0231253,
        std_adc = 52.4672,
        name='CanonT1'),
    'SonyA7r3': CameraParams(
        k=[2.23516, 1.45227, 5.63239],
        std_read = 3.36703e-07,
        std_adc = 7.68687,
        name='SonyA7r3'),
    'IDS': CameraParams(
        k=[13.6752, 9.88293, 8.90192],
        std_read = 0.570401,
        std_adc = 1.3521,
        name='IDS')
}

