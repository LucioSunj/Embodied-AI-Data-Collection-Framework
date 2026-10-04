import ctypes
import numpy as np
from scipy.spatial.transform import Rotation as R


class Omega7Interface:
    def __init__(self, lib_path="libdhd.so"):
        self.lib = ctypes.CDLL(lib_path)

        # SDK function signatures from dhdc.h
        self.lib.dhdOpen.argtypes = []
        self.lib.dhdOpen.restype = ctypes.c_int
        self.lib.dhdClose.argtypes = [ctypes.c_char]
        self.lib.dhdClose.restype = ctypes.c_int
        self.lib.dhdEnableForce.argtypes = [ctypes.c_ubyte, ctypes.c_char]
        self.lib.dhdEnableForce.restype = ctypes.c_int
        self.lib.dhdGetPositionAndOrientationFrame.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_char,
        ]
        self.lib.dhdGetPositionAndOrientationFrame.restype = ctypes.c_int
        self.lib.dhdGetGripperAngleDeg.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_char,
        ]
        self.lib.dhdGetGripperAngleDeg.restype = ctypes.c_int

        self.dev_id = self.lib.dhdOpen()
        if self.dev_id < 0:
            raise RuntimeError("握手失败")
        self.lib.dhdEnableForce(1, self.dev_id)
        print(f"✅ 成功加载底层驱动: {lib_path}")
        print(f"🚀 Omega.7 底层通讯已握手 (Device ID: {self.dev_id})")

    def get_state(self):
        px, py, pz = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
        c_matrix = ((ctypes.c_double * 3) * 3)()

        err = self.lib.dhdGetPositionAndOrientationFrame(
            ctypes.byref(px),
            ctypes.byref(py),
            ctypes.byref(pz),
            ctypes.cast(c_matrix, ctypes.POINTER(ctypes.c_double)),
            self.dev_id,
        )
        if err < 0:
            raise RuntimeError("读取 Omega 位姿失败")

        g_angle = ctypes.c_double()
        err = self.lib.dhdGetGripperAngleDeg(ctypes.byref(g_angle), self.dev_id)
        if err < 0:
            raise RuntimeError("读取 Omega 夹爪角度失败")

        pos = np.array([px.value, py.value, pz.value], dtype=float)
        rot_matrix = np.array(c_matrix, dtype=float)
        quat = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]

        return pos, quat, g_angle.value

    def close(self):
        self.lib.dhdClose(self.dev_id)
