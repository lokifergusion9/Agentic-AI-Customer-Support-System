"""Typed results of the order read service (W1.11)."""

from datetime import datetime

from pydantic import BaseModel


class OrderItemView(BaseModel):
    product_id: str
    product_name: str
    quantity: int
    # Line total in minor units (paise).
    paid_amount_minor: int


class OrderView(BaseModel):
    id: str
    status: str
    currency: str
    # Includes shipping.
    paid_amount_minor: int
    shipping_amount_minor: int
    placed_at: datetime
    delivered_at: datetime | None
    items: list[OrderItemView]


class OrderSummary(BaseModel):
    """One line of a customer's order list, used to ask which order they mean (W2.13)."""

    id: str
    status: str
    currency: str
    paid_amount_minor: int
    placed_at: datetime
    delivered_at: datetime | None
    product_names: list[str]
