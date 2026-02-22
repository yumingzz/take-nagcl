"""
GPT-2 shift 事件回答生成器

讀取 knowSelect 推論輸出（metrics/shift_top3.jsonl），對每個 shift 事件生成回答，
並將「結構化資訊 + GPT-2 生成內容」寫入文字檔。

設計目標：
- 不依賴 knowSelect 訓練流程內部狀態，純讀檔再產生輸出
- 輸出可回溯：turn_id 以 tiage_anno_nodes_all.csv 對照為準（由上游已寫入 shift_top3.jsonl）
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class GeneratorConfig:
    model_name_or_path: str = "gpt2"
    max_new_tokens: int = 80
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _fmt_top3(items: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, it in enumerate(items[:3], start=1):
        lines.append(
            f"  [{i}] turn_id={it.get('turn_id', 'N/A')}  centrality={it.get('centrality', 0):.6f}\n"
            f"      sentence={it.get('sentence', '').strip()}\n"
        )
    return "".join(lines).rstrip() + ("\n" if lines else "")


def _build_prompt(shift_sentence: str, interval_top3: List[Dict[str, Any]]) -> str:
    # 以「轉移句 + 區間 Top-3」做為 prompt，上下文足夠且易讀
    top3_text = "\n".join([f"- {it.get('sentence', '').strip()}" for it in interval_top3[:3] if it.get("sentence")])
    prompt = (
        "以下是一段對話在主題轉移時的關鍵資訊，請生成一則自然、連貫的回覆：\n"
        f"【發生轉移的句子】{shift_sentence.strip()}\n"
        "【區間內中心性 Top-3 句子】\n"
        f"{top3_text}\n"
        "【回覆】"
    )
    return prompt


def generate_shift_answers_txt(
    shift_top3_jsonl: str,
    out_txt: str,
    cfg: Optional[GeneratorConfig] = None,
    only_split: Optional[str] = None,
    only_epoch: Optional[str] = None,
) -> None:
    """
    讀取 shift_top3.jsonl，對每個 shift 事件生成回答並寫入 out_txt（覆寫）。

    - only_split / only_epoch：可選過濾（對應 jsonl record_out 的 dataset/epoch；其中 dataset 實際上是 split 名稱，如 test）
    """
    cfg = cfg or GeneratorConfig()

    # 延後 import，避免在不需要生成時強制載入 transformers/torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer  # type: ignore
    import torch  # type: ignore

    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name_or_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as out:
        out.write("=== shift 事件 GPT-2 回答 ===\n")
        out.write(f"model_name_or_path: {cfg.model_name_or_path}\n")
        out.write(f"max_new_tokens: {cfg.max_new_tokens}\n")
        out.write(f"temperature: {cfg.temperature}\n")
        out.write(f"top_p: {cfg.top_p}\n")
        out.write("\n")

        for rec in _iter_jsonl(shift_top3_jsonl):
            split = str(rec.get("dataset", ""))
            epoch = str(rec.get("epoch", ""))
            if only_split is not None and split != only_split:
                continue
            if only_epoch is not None and only_epoch != "all" and epoch != only_epoch:
                continue

            dialog_id = rec.get("dialog_id", "N/A")
            shift_events: List[Dict[str, Any]] = rec.get("shift_events") or []
            if not shift_events:
                continue

            out.write(f"--- dialog_id={dialog_id}  split={split}  epoch={epoch} ---\n")
            for idx, ev in enumerate(shift_events, start=1):
                shift_sentence = str(ev.get("shift_sentence", "")).strip()
                shift_turn_id = ev.get("shift_turn_id", "N/A")
                shift_centrality = float(ev.get("shift_centrality", 0.0))
                interval_top3: List[Dict[str, Any]] = ev.get("interval_top3") or []

                # 明確寫出你已確認的區間規則（避免誤會）
                out.write(f"\n[ShiftEvent {idx}]\n")
                out.write("區間規則：包含本次 shift（含端點），不包含前一次 shift\n")
                out.write(f"shift_turn_id={shift_turn_id}\n")
                out.write(f"shift_centrality={shift_centrality:.6f}\n")
                out.write(f"shift_sentence={shift_sentence}\n")
                out.write("\n[IntervalTop3]\n")
                out.write(_fmt_top3(interval_top3))

                prompt = _build_prompt(shift_sentence, interval_top3)
                inputs = tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=cfg.max_new_tokens,
                        do_sample=cfg.do_sample,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # 只取「【回覆】」後的生成內容（盡量避免重複前文）
                reply = gen_text.split("【回覆】", 1)[-1].strip()
                out.write("\n[GPT2Reply]\n")
                out.write(reply + "\n")

            out.write("\n")
