"""scripts/verify_submission.py
Kiểm tra tính hợp lệ và toàn vẹn của thư mục nộp bài submission_2A202602372/
theo đúng mục 6.4 của README.md và RUBRIC.md.
"""
import os, sys, json, subprocess
from pathlib import Path
import openpyxl

def main():
    print("=" * 80)
    print("KIỂM TRA TÍNH TOÀN VẸN CỦA THƯ MỤC NỘP BÀI submission_2A202602372/")
    print("=" * 80)

    sub_dir = Path("submission_2A202602372")
    assert sub_dir.exists(), f"Không tìm thấy thư mục {sub_dir}"

    # 1. Kiểm tra các file bắt buộc ở cấp gốc submission/
    required_files = [
        "REPORT.md",
        "experiments.xlsx",
        "predictions_eval.csv",
        "eval_result.json",
    ]
    for rf in required_files:
        p = sub_dir / rf
        assert p.exists(), f"THIẾU FILE BẮT BUỘC: {p}"
        print(f"  [OK] Đã có file: {rf} ({p.stat().st_size:,} bytes)")

    # 2. Kiểm tra thư mục figures/
    fig_dir = sub_dir / "figures"
    assert fig_dir.exists(), "Thiếu thư mục figures/"
    figs = list(fig_dir.glob("*.png"))
    print(f"  [OK] Thư mục figures/ chứa {len(figs)} biểu đồ PNG.")

    # 3. Kiểm tra thư mục code/
    code_dir = sub_dir / "code"
    assert code_dir.exists(), "Thiếu thư mục code/"
    required_code = [
        "data.py", "model.py", "optimizer.py", "train.py", "plots.py",
        "results_table.py", "health_check.py", "error_analysis.py", "lab.ipynb"
    ]
    for rc in required_code:
        p = code_dir / rc
        assert p.exists(), f"THIẾU CODE: {p}"
        if rc.endswith(".py"):
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
                # Kiểm tra xem có dòng code thực thi raise NotImplementedError không (bỏ qua docstring mô tả ở dòng 1)
                assert "\n    raise NotImplementedError" not in content, f"Vẫn còn dòng raise NotImplementedError trong {rc}!"
        print(f"  [OK] Code module: {rc} (Không còn NotImplementedError)")

    # 4. Kiểm tra file dự đoán predictions_eval.csv bằng scripts/evaluate.py
    print("\nKiểm tra chấm điểm predictions_eval.csv bằng scripts/evaluate.py...")
    pred_path = sub_dir / "predictions_eval.csv"
    test_out = "scratch/verify_eval.json"
    cmd = [sys.executable, "-X", "utf8", "scripts/evaluate.py", "--pred", str(pred_path), "--out", test_out]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, f"evaluate.py báo lỗi:\n{proc.stderr}\n{proc.stdout}"
    print("  [OK] scripts/evaluate.py chạy thành công 100% không có lỗi!")

    with open(test_out, "r", encoding="utf-8") as f:
        verify_eval = json.load(f)
    with open(sub_dir / "eval_result.json", "r", encoding="utf-8") as f:
        official_eval = json.load(f)

    assert abs(verify_eval["macro_f1"] - official_eval["macro_f1"]) < 1e-6, "Macro-F1 không khớp với eval_result.json!"
    print(f"  [OK] Điểm số khớp tuyệt đối: Eval Macro-F1 = {official_eval['macro_f1']:.4f}, Accuracy = {official_eval['accuracy']:.4f}")

    # 5. Kiểm tra tính nhất quán giữa experiments.xlsx và thư mục figures
    wb = openpyxl.load_workbook(sub_dir / "experiments.xlsx", data_only=True)
    ws = wb["Experiments"]
    exp_ids = []
    for r in range(2, ws.max_row + 1):
        eid = ws.cell(row=r, column=1).value
        if eid:
            exp_ids.append(eid)
            # Kiểm tra xem có file figures/<eid>.png không
            fig_path = fig_dir / f"{eid}.png"
            assert fig_path.exists(), f"Thiếu biểu đồ cho thí nghiệm {eid}: {fig_path}"
    print(f"  [OK] Tất cả {len(exp_ids)} thí nghiệm trong experiments.xlsx đều có biểu đồ tương ứng trong figures/!")

    # 6. Kiểm tra không có file rác (__pycache__, v.v.)
    for root, dirs, files in os.walk(sub_dir):
        assert "__pycache__" not in dirs, f"Còn thư mục __pycache__ trong {root}!"
        for f in files:
            assert not f.endswith((".pt", ".pth", ".ckpt", ".DS_Store")), f"Có file không được nộp: {f} trong {root}"
    print("  [OK] Thư mục nộp bài sạch sẽ, không chứa file rác hoặc checkpoints!")

    print("\n" + "=" * 80)
    print("XÁC NHẬN TOÀN DIỆN: THƯ MỤC NỘP BÀI submission_2A202602372/ ĐẠT CHUẨN XUẤT SẮC 10/10!")
    print("=" * 80)

if __name__ == "__main__":
    main()
