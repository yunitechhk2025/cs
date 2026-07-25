"""脱敏版专用：知识库原文里的真实品牌名和真人姓名，在加载时统一替换掉。

两个产品的知识来源（Excel 题库、产品说明文档）都是直接从正式版拿过来的原始资料，里面写着
真实品牌名和创始人姓名；而这套系统给客户的回答是把题库答案**原文照搬**的（专业内容不经 AI
改写，确保逐字一致），所以只要不在源头替换，脱敏版的客户就会在回答里看到真实品牌。

替换做在"加载知识库"这一层，而不是直接改 Excel/文档文件本身：原始资料保持与正式版一致，
以后题库更新时直接替换文件即可，不用每次重新做一遍脱敏处理。

替换按下面的先后顺序执行，长的规则必须排在短的前面——否则"姜建华医生"会先被"姜"开头的
短规则改坏，剩下"建华"这种半截结果。
"""

import re

REPLACEMENTS = (
    ("姜建华医生", "yuni医生"),
    ("姜建華醫生", "yuni医生"),
    ("姜建华", "yuni医生"),
    ("姜医生", "yuni医生"),
    ("姜醫生", "yuni医生"),
    ("澳洲肤润康", "YUNI"),
    ("澳洲膚潤康", "YUNI"),
    ("肤润康", "YUNI"),
    ("膚潤康", "YUNI"),
    ("富润康", "YUNI"),
    # 产品说明文档里还写了英文品牌名（"澳洲肤润康（FUYUNHON）10% 碳酰二胺护手乳霜"）
    ("FUYUNHON", "YUNI"),
    ("Fuyunhon", "YUNI"),
    ("fuyunhon", "YUNI"),
    # 收尾：原文里"中文品牌名（英文品牌名）"这种写法，两个名字都换成 YUNI 之后会变成
    # "YUNI（YUNI）"这种读起来很奇怪的重复，这里合并掉。必须排在上面的品牌规则之后。
    ("YUNI（YUNI）", "YUNI"),
    ("YUNI(YUNI)", "YUNI"),
)


# 替换完成后，中文品牌名紧跟数字的地方会粘成"YUNI10%碳酰二胺…"（原文是"澳洲肤润康 10%…"，
# 中文和数字之间本来不需要空格，换成英文品牌名后就需要了），这里补一个空格。
_SPACE_BEFORE_DIGIT = re.compile(r"YUNI(?=\d)")


def desensitize(text: str) -> str:
    """把知识库原文里的真实品牌名/姓名替换成脱敏后的名称。非字符串原样返回。"""
    if not isinstance(text, str) or not text:
        return text
    for old, new in REPLACEMENTS:
        if old in text:
            text = text.replace(old, new)
    return _SPACE_BEFORE_DIGIT.sub("YUNI ", text)
