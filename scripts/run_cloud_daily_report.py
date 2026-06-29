from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from trading_day import is_trading_day, load_holidays, previous_trading_day


ROOT = Path(__file__).absolute().parents[1]
MODEL_DIRS = [
    "models/curve_2020_AAA3Y_h5/01_full_features/gru",
    "models/curve_2020_AAA3Y_h5/01_full_features/tcn",
    "models/curve_2020_AAA3Y_h5/01_full_features/transformer",
    "models/curve_2020_AAA5Y_h5/01_full_features/gru",
    "models/curve_2020_AAA5Y_h5/01_full_features/tcn",
    "models/curve_2020_AAA5Y_h5/01_full_features/transformer",
    "models/curve_2020_AAA10Y_h5/01_full_features/gru",
    "models/curve_2020_AAA10Y_h5/01_full_features/tcn",
    "models/curve_2020_AAA10Y_h5/01_full_features/transformer",
    "models/curve_2020_AAAp20Y_h5/01_full_features/gru",
    "models/curve_2020_AAAp20Y_h5/01_full_features/tcn",
    "models/curve_2020_AAAp20Y_h5/01_full_features/transformer",
]
REQUIRED_TENORS = ("3年", "5年", "10年", "20年")


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cloud-friendly daily credit-bond AI report pipeline.")
    parser.add_argument("--today", default=date.today().isoformat())
    parser.add_argument("--same-day", action="store_true", help="Use today's date as data end. Default is previous trading day.")
    parser.add_argument("--holiday-file", default="configs/china_market_holidays.csv")
    parser.add_argument("--out-dir", default="data/dm_daily_master_curve_2020")
    parser.add_argument("--site-dir", default="output/site")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--full-refresh", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-email", action="store_true")
    parser.add_argument("--dry-run-email", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("运行：", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def validate_report_has_model_data(json_report: Path) -> None:
    report = json.loads(json_report.read_text(encoding="utf-8"))
    tenors = {
        str(item.get("label", "")): item
        for item in (report.get("market_snapshot") or {}).get("tenors", [])
    }
    issues: list[str] = []
    for label in REQUIRED_TENORS:
        tenor = tenors.get(label)
        if not tenor:
            issues.append(f"{label} 期限没有出现在 market_snapshot。")
            continue
        if not tenor.get("available"):
            issues.append(f"{label} 期限没有可用收益率曲线数据。")
        ensemble = tenor.get("ensemble_prediction") or {}
        if int(ensemble.get("model_count") or 0) <= 0:
            issues.append(f"{label} 期限没有可用模型预测结果。")

    ok_predictions = [item for item in report.get("predictions", []) if item.get("ok")]
    if not ok_predictions:
        issues.append("全部模型预测均失败。")

    if not issues:
        print(f"日报数据校验通过：{len(ok_predictions)} 个模型预测可用，四个期限曲线均可用。")
        return

    print("日报数据校验失败，以下是关键诊断信息：")
    print(f"- 特征形状：{report.get('features_shape')}")
    print(f"- 数据范围：{report.get('data_start')} 至 {report.get('data_end')}")
    print(f"- 可用模型预测数：{len(ok_predictions)} / {len(report.get('predictions', []))}")
    curve_rows = [row for row in report.get("fetch", []) if row.get("source") == "curve_func"]
    print(f"- 曲线抓取记录数：{len(curve_rows)}")
    for row in curve_rows:
        print(
            "  曲线："
            f"{row.get('curve_name')} term={row.get('curve_term')} "
            f"new_rows={row.get('new_rows')} total_rows={row.get('total_rows')} "
            f"file={row.get('output_file')}"
        )
    if not curve_rows:
        print("- 没有任何 curve_func 抓取记录，请检查配置是否被云端正确读取。")
    for item in report.get("predictions", []):
        if not item.get("ok"):
            print(f"  模型失败：{item.get('model_dir')} -> {item.get('error')}")
    raise RuntimeError("日报缺少曲线或模型数据，已停止发送空日报：" + "；".join(issues))


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    today = date.fromisoformat(args.today)
    holidays = load_holidays(args.holiday_file)
    if not is_trading_day(today, holidays):
        print(f"{today.isoformat()} 不是交易日，跳过日报。")
        return 0

    data_end = today if args.same_day else previous_trading_day(today, holidays)
    out_dir = Path(args.out_dir)
    report_dir = out_dir / "reports"
    json_report = report_dir / "daily_dm_update_report.json"
    html_report = report_dir / "daily_dm_update_report.html"
    pdf_path = ROOT / "output" / "daily_reports" / f"creditbond_ai_daily_{data_end.isoformat()}.pdf"

    daily_cmd = [
        sys.executable,
        "-m",
        "creditbond_ai.cli",
        "daily-dm-update",
        "--config",
        "configs/dm_master_indicators.csv",
        "--out-dir",
        str(out_dir),
        "--end-date",
        data_end.isoformat(),
        "--model-ready-max-missing-ratio",
        "0.2",
        "--device",
        args.device,
        "--timeout",
        "60",
    ]
    if args.start_date:
        daily_cmd += ["--start-date", args.start_date]
    if args.full_refresh:
        daily_cmd.append("--full-refresh")
    if args.skip_fetch:
        daily_cmd.append("--skip-fetch")
    for model_dir in MODEL_DIRS:
        daily_cmd += ["--model-dir", model_dir]
    run(daily_cmd)
    validate_report_has_model_data(json_report)

    run(
        [
            sys.executable,
            "scripts/build_daily_report_pdf.py",
            "--json-report",
            str(json_report),
            "--html-report",
            str(html_report),
            "--out",
            str(pdf_path),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/publish_daily_site.py",
            "--json-report",
            str(json_report),
            "--html-report",
            str(html_report),
            "--pdf",
            str(pdf_path),
            "--out-dir",
            args.site_dir,
        ]
    )

    if not args.skip_email:
        email_cmd = [
            sys.executable,
            "scripts/send_daily_report_resend.py",
            "--to",
            os.environ.get("RESEND_TO") or "yuweiqian@cib.com.cn",
            "--json-report",
            str(json_report),
            "--report",
            str(html_report),
            "--pdf",
            str(pdf_path),
        ]
        if args.dry_run_email:
            email_cmd.append("--dry-run")
        run(email_cmd)

    print(f"完成：HTML={html_report} PDF={pdf_path} SITE={ROOT / args.site_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
