#
#  Cipher/PKCS1_OAEP.py : PKCS#1 OAEP
#
# ===================================================================
# The contents of this file are dedicated to the public domain.  To
# the extent that dedication to the public domain is not available,
# everyone is granted a worldwide, perpetual, royalty-free,
# non-exclusive license to exercise all rights associated with the
# contents of this file for any purpose whatsoever.
# No rights are reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================

from hashlib import sha1

import Crypto.Hash.SHA1
import Crypto.Util.number
import gmpy2  # type: ignore
from Crypto import Random
from Crypto.Signature.pss import MGF1
from Crypto.Util.number import bytes_to_long, ceil_div, long_to_bytes
from Crypto.Util.py3compat import _copy_bytes, bord
from Crypto.Util.strxor import strxor


class PKCS1OAepCipher:
    """
    PKCS#1 v1.5 OAEP 加密方案实现类

    作用:
        实现RSA-OAEP加密标准（RFC 3447），提供消息填充、加密及解密功能
        支持自定义哈希算法和掩码生成函数（MGF）

    注意:
        禁止直接实例化，应通过工厂方法 :func:`new` 创建对象
        发送方与接收方需使用相同的哈希算法和MGF函数
    """

    def __init__(self, key, hashAlgo, mgfunc, label, randfunc):
        """
        初始化OAEP加密对象

        参数:
            key : RSA密钥对象
                包含公钥时可加密，包含私钥时可解密
            hashAlgo : 哈希算法对象
                指定使用的哈希函数（如SHA256），默认SHA1
            mgfunc : 可调用对象
                掩码生成函数，默认使用与hashAlgo匹配的MGF1
            label : bytes/bytearray/memoryview
                加密标签（不影响安全性），默认空字节
            randfunc : 可调用对象
                安全随机字节生成函数，必须符合密码学安全要求

        安全建议:
            修改MGF函数需充分理解其影响，否则应保持默认
        """
        self._key = key

        # 初始化哈希算法（默认SHA1）
        self._hashObj = hashAlgo if hashAlgo else Crypto.Hash.SHA1

        # 初始化掩码生成函数（默认MGF1）
        self._mgf = mgfunc if mgfunc else lambda x, y: MGF1(x, y, self._hashObj)

        # 处理标签（防御性拷贝）
        self._label = _copy_bytes(None, None, label)
        # 随机数生成器（必须为密码学安全源）
        self._randfunc = randfunc

    def can_encrypt(self):
        """遗留方法-检查是否支持加密（已废弃，仅保持向后兼容）"""
        return self._key.can_encrypt()

    def can_decrypt(self):
        """遗留方法-检查是否支持解密（已废弃，仅保持向后兼容）"""
        return self._key.can_decrypt()

    def encrypt(self, message):
        """
        OAEP加密方法

        参数:
            message : 待加密明文（字节数据）
                最大长度 = RSA模数字节长度 - 2*hash长度 - 2
                例如2048位RSA + SHA256 => 256 - 32*2 -2 = 190字节

        返回:
            bytes : 密文字节数据，长度等于RSA模数字节长度

        异常:
            ValueError : 明文长度超过限制时抛出
        """
        # 步骤参考RFC3447 7.1.1节
        modBits = Crypto.Util.number.size(self._key.n)  # 模数位数
        k = ceil_div(modBits, 8)  # 字节长度计算
        hLen = self._hashObj.digest_size  # 哈希长度
        mLen = len(message)  # 明文长度

        # 步骤1b：检查明文长度
        ps_len = k - mLen - 2 * hLen - 2
        if ps_len < 0:
            raise ValueError("明文过长，最大允许长度: {}字节".format(k - 2 * hLen - 2))

        # 步骤2a：生成标签哈希
        lHash = sha1(self._label).digest()  # 使用SHA1处理标签

        # 步骤2b-2c：构造数据块DB
        ps = b"\x00" * ps_len  # 填充字节
        db = lHash + ps + b"\x01" + _copy_bytes(None, None, message)  # 拼接数据块

        # 步骤2d-2h：生成掩码
        ros = self._randfunc(hLen)  # 随机种子
        dbMask = self._mgf(ros, k - hLen - 1)  # 数据块掩码
        maskedDB = strxor(db, dbMask)  # 异或操作
        seedMask = self._mgf(maskedDB, hLen)  # 种子掩码
        maskedSeed = strxor(ros, seedMask)  # 掩码种子

        # 步骤2i：构造编码消息
        em = b"\x00" + maskedSeed + maskedDB

        # 步骤3a-3c：RSA加密
        em_int = bytes_to_long(em)  # 字节转大整数
        m_int = gmpy2.powmod(em_int, self._key.e, self._key.n)  # 模幂运算
        c = long_to_bytes(m_int, k)  # 转换回字节

        return c

    def decrypt(self, ciphertext):
        """
        OAEP解密方法

        参数:
            ciphertext : 待解密密文（字节数据）
                长度必须等于RSA模数字节长度

        返回:
            bytes : 解密后的原始明文

        异常:
            ValueError : 密文长度错误或完整性校验失败
            TypeError : 使用公钥尝试解密时抛出
        """
        # 步骤参考RFC3447 7.1.2节
        modBits = Crypto.Util.number.size(self._key.n)
        k = ceil_div(modBits, 8)
        hLen = self._hashObj.digest_size

        # 步骤1b-1c：密文长度校验
        if len(ciphertext) != k or k < hLen + 2:
            raise ValueError("无效密文长度，预期长度: {}字节".format(k))

        # 步骤2a-2b：RSA解密
        c_int = bytes_to_long(ciphertext)
        m_int = pow(c_int, self._key.d, self._key.n)  # 使用私钥指数
        em = long_to_bytes(m_int, k)  # 转换回字节

        # 步骤2c：分解编码消息
        _, maskedSeed, maskedDB = em[:1], em[1:hLen + 1], em[hLen + 1:]

        # 步骤2d-2e：恢复种子和数据块
        seedMask = self._mgf(maskedDB, hLen)
        ros = strxor(maskedSeed, seedMask)
        dbMask = self._mgf(ros, k - hLen - 1)
        db = strxor(maskedDB, dbMask)

        # 步骤3a：分离哈希、填充和明文
        lHash_prime = db[:hLen]
        # 查找填充结束分隔符0x01
        sep_index = db[hLen:].find(b'\x01') + hLen
        if sep_index < hLen:
            raise ValueError("解密失败：无效填充结构")

        # 步骤3b：标签哈希校验
        if lHash_prime != sha1(self._label).digest():
            raise ValueError("解密失败：标签哈希不匹配")

        # 提取明文
        message = db[sep_index + 1:]
        return message
