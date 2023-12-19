config = {
    "input": {
        "train": "../review/open/train.csv",
        "test": "../review/open/test.csv",
    },
    "columns": {
        "use_trn_cols": ["ID", "사고일시", "요일", "기상상태", "시군구", "도로형태", "노면상태", "사고유형"],
        "drop_trn_flag": True,
        "drop_trn_cols": [
            "사고유형 - 세부분류",
            "법규위반",
            "가해운전자 차종",
            "가해운전자 성별",
            "가해운전자 연령",
            "가해운전자 상해정도",
            "피해운전자 차종",
            "피해운전자 성별",
            "피해운전자 연령",
            "피해운전자 상해정도",
            "사망자수",
            "중상자수",
            "경상자수",
            "부상자수",
        ],
        "use_tst_cols": ["ID", "사고일시", "요일", "기상상태", "시군구", "도로형태", "노면상태", "사고유형"],
        "drop_tst_flag": False,
        "drop_tst_cols": [],
    },
}
