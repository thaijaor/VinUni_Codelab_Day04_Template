import json
import os
from typing import List, Dict, Any
from datetime import datetime

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw-data")

VALID_CATEGORIES = ("xe_dien", "du_lich")
VALID_PRIORITIES = ("low", "medium", "high")

# ---------------------------------------------------------------------------
# Tool #1: search_product_catalog
# ---------------------------------------------------------------------------

def search_product_catalog(category: str, max_price: int = 999999999999) -> List[Dict[str, Any]]:
    """
    Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục và giá tối đa.

    Args:
        category: Loại sản phẩm ('xe_dien' hoặc 'du_lich').
        max_price: Giá tối đa (VNĐ). Mặc định không giới hạn.

    Returns:
        Danh sách sản phẩm phù hợp điều kiện.
    """
    catalog_file = os.path.join(RAW_DATA_DIR, "product_catalog.json")
    if not os.path.exists(catalog_file):
        return [{"error": "Product catalog file not found."}]
    if category.lower() not in VALID_CATEGORIES:
        return [{"error": f"Invalid category '{category}'. Use one of {list(VALID_CATEGORIES)}."}]

    with open(catalog_file, "r", encoding="utf-8") as f:
        products = json.load(f)

    results = [
        p for p in products
        if p["category"].lower() == category.lower()
        and p["price_vnd"] <= max_price
    ]
    return sorted(results, key=lambda p: p["price_vnd"])


# ---------------------------------------------------------------------------
# Tool #2: submit_support_ticket
# ---------------------------------------------------------------------------

def submit_support_ticket(
    customer_name: str,
    issue_description: str,
    priority: str = "medium"
) -> Dict[str, Any]:
    """
    Ghi nhận yêu cầu hỗ trợ của khách hàng vào hệ thống ticket.

    Args:
        customer_name: Tên khách hàng.
        issue_description: Mô tả vấn đề cần hỗ trợ.
        priority: Mức độ ưu tiên ('low', 'medium', 'high'). Mặc định 'medium'.

    Returns:
        Thông tin ticket vừa tạo bao gồm ticket_id, status.
    """
    tickets_file = os.path.join(RAW_DATA_DIR, "support_tickets.json")
    priority = priority.lower()
    if priority not in VALID_PRIORITIES:
        priority = "medium"

    # Load trước rồi append — tránh ghi đè ticket cũ
    existing_tickets = []
    if os.path.exists(tickets_file):
        with open(tickets_file, "r", encoding="utf-8") as f:
            existing_tickets = json.load(f)

    today = datetime.now().strftime("%Y%m%d")
    seq = len(existing_tickets) + 1
    ticket_id = f"TK-{today}-{seq:03d}"

    new_ticket = {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "issue_description": issue_description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now().isoformat() + "+07:00",
        "category": "general"
    }
    existing_tickets.append(new_ticket)

    with open(tickets_file, "w", encoding="utf-8") as f:
        json.dump(existing_tickets, f, indent=2, ensure_ascii=False)

    return {
        "ticket_id": ticket_id,
        "customer_name": customer_name,
        "priority": priority,
        "status": "open",
        "message": f"Ticket {ticket_id} đã được tạo thành công."
    }


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — JSON Schemas mô tả cho LLM
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_product_catalog",
        "description": (
            "Tra cứu sản phẩm/dịch vụ Vingroup (xe điện VinFast, gói nghỉ dưỡng Vinpearl) "
            "theo danh mục và giá tối đa. Dùng khi khách muốn xem, tìm, so sánh sản phẩm hoặc hỏi giá. "
            "KHÔNG dùng cho câu hỏi chính sách chung hoặc khiếu nại."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Loại sản phẩm: 'xe_dien' (VinFast) hoặc 'du_lich' (Vinpearl).",
                    "enum": ["xe_dien", "du_lich"]
                },
                "max_price": {
                    "type": "integer",
                    "description": "Giá tối đa tính bằng VNĐ, ví dụ 'dưới 600 triệu' -> 600000000. Bỏ trống nếu khách không nêu giá."
                }
            },
            "required": ["category"]
        }
    },
    {
        "name": "submit_support_ticket",
        "description": (
            "Tạo ticket hỗ trợ khi khách báo lỗi, sự cố, khiếu nại hoặc muốn ghi nhận phản hồi về sản phẩm/dịch vụ Vingroup. "
            "Chỉ gọi khi đã biết tên khách hàng và mô tả vấn đề; nếu thiếu thì hỏi lại khách."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "Họ tên đầy đủ của khách hàng, ví dụ 'Lê Minh Khoa'."
                },
                "issue_description": {
                    "type": "string",
                    "description": "Mô tả ngắn gọn vấn đề, ví dụ 'Xe VF 8 bị lỗi hệ thống ADAS'."
                },
                "priority": {
                    "type": "string",
                    "description": "Mức độ ưu tiên: 'high' (gấp/nghiêm trọng), 'medium' (trung bình, mặc định), 'low' (không gấp).",
                    "enum": ["low", "medium", "high"]
                }
            },
            "required": ["customer_name", "issue_description"]
        }
    }
]


# ---------------------------------------------------------------------------
# TOOL_MAP — Ánh xạ tên tool → hàm thực thi
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "search_product_catalog": search_product_catalog,
    "submit_support_ticket": submit_support_ticket
}
