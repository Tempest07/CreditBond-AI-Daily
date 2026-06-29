LABEL_NAMES = {
    0: "bearish",
    1: "bullish",
    2: "range",
}

LABEL_NAMES_CN = {
    0: "看空",
    1: "看多",
    2: "震荡",
}

ADVICE_CN = {
    0: "建议降低信用债风险暴露或等待更好的买点",
    1: "建议提高信用债风险暴露或维持积极持仓",
    2: "建议观望或维持现有持仓",
}


def label_to_signal(label: int) -> int:
    """Map class label to a directional credit-bond position."""
    if label == 1:
        return 1
    if label == 0:
        return -1
    return 0
