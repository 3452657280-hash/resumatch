"""历史分析记录服务 —— 基于 SQLite 存储匹配/批量分析结果。"""
from __future__ import annotations
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.models import MatchReport, HistoryRecord

HISTORY_DB = str(Path(settings.chroma_persist_dir).parent / "history.db")


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(HISTORY_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY,
            jd_text TEXT NOT NULL,
            match_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            report_json TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


class HistoryService:
    """历史记录管理。"""

    @staticmethod
    def save(
        jd_text: str,
        match_type: str,
        report: MatchReport,
    ) -> HistoryRecord:
        """保存一条分析记录。"""
        record_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        conn = _get_db()
        conn.execute(
            "INSERT INTO history (id, jd_text, match_type, created_at, report_json) VALUES (?, ?, ?, ?, ?)",
            (
                record_id,
                jd_text,
                match_type,
                now,
                report.model_dump_json(),
            ),
        )
        conn.commit()

        return HistoryRecord(
            id=record_id,
            jd_text=jd_text[:100],
            match_type=match_type,
            created_at=now,
            report=report,
        )

    @staticmethod
    def list(page: int = 1, page_size: int = 20) -> list[dict[str, Any]]:
        """列出历史记录摘要（含评分预览）。"""
        conn = _get_db()
        rows = conn.execute(
            "SELECT id, jd_text, match_type, created_at, report_json FROM history ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (page_size, (page - 1) * page_size),
        ).fetchall()
        records = []
        for r in rows:
            rec = dict(r)
            report_json = rec.pop("report_json", None)
            if report_json:
                try:
                    report_data = json.loads(report_json)
                    scores = [res.get("overall_score", 0) for res in report_data.get("results", [])]
                    rec["score_count"] = len(scores)
                    rec["top_scores"] = sorted(scores, reverse=True)[:5]
                    rec["avg_score"] = round(sum(scores) / len(scores)) if scores else 0
                except (json.JSONDecodeError, KeyError):
                    rec["score_count"] = 0
                    rec["top_scores"] = []
                    rec["avg_score"] = 0
            else:
                rec["score_count"] = 0
                rec["top_scores"] = []
                rec["avg_score"] = 0
            records.append(rec)
        return records

    @staticmethod
    def get(record_id: str) -> HistoryRecord | None:
        """获取单条历史记录的完整数据。"""
        conn = _get_db()
        row = conn.execute(
            "SELECT * FROM history WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            return None
        report = MatchReport.model_validate_json(row["report_json"])
        return HistoryRecord(
            id=row["id"],
            jd_text=row["jd_text"],
            match_type=row["match_type"],
            created_at=row["created_at"],
            report=report,
        )

    @staticmethod
    def delete(record_id: str) -> bool:
        """删除一条历史记录。"""
        conn = _get_db()
        cursor = conn.execute("DELETE FROM history WHERE id = ?", (record_id,))
        conn.commit()
        return cursor.rowcount > 0

    @staticmethod
    def batch_delete(record_ids: list[str]) -> int:
        """批量删除历史记录。"""
        conn = _get_db()
        placeholders = ",".join(["?" for _ in record_ids])
        cursor = conn.execute(f"DELETE FROM history WHERE id IN ({placeholders})", record_ids)
        conn.commit()
        return cursor.rowcount

    @staticmethod
    def count() -> int:
        conn = _get_db()
        return conn.execute("SELECT COUNT(*) as c FROM history").fetchone()["c"]
