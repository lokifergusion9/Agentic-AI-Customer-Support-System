"""Order read service (W1.11): orders are read from the seeded tables, not a separate API.

Ownership is checked here, in the service. An order ID from the customer or the model does
not grant access: a missing order and someone else's order fail the same way.
"""

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.db import SCHEMA
from app.schemas.orders import OrderItemView, OrderSummary, OrderView
from app.services.errors import ServiceFailure

ORDER_COLUMNS = (
    "id, status, currency, paid_amount_minor, shipping_amount_minor, placed_at, delivered_at"
)


def order_not_found() -> ServiceFailure:
    return ServiceFailure("ORDER_NOT_FOUND", "No order with this ID was found.")


def get_order(db: Session, customer: AuthenticatedCustomer, order_id: str) -> OrderView:
    order = db.execute(
        text(
            f"SELECT {ORDER_COLUMNS} FROM {SCHEMA}.orders "
            "WHERE id = :order AND customer_id = :customer"
        ),
        {"order": order_id, "customer": customer.customer_id},
    ).one_or_none()
    if order is None:
        raise order_not_found()

    items = db.execute(
        text(
            "SELECT product_id, product_name, quantity, paid_amount_minor "
            f"FROM {SCHEMA}.order_items WHERE order_id = :order ORDER BY id"
        ),
        {"order": order_id},
    ).all()
    return OrderView(
        **order._mapping,
        items=[OrderItemView.model_validate(item._mapping) for item in items],
    )


def list_orders(
    db: Session, customer: AuthenticatedCustomer, *, limit: int = 10
) -> list[OrderSummary]:
    """The customer's own orders, newest first."""
    rows = db.execute(
        text(
            "SELECT o.id, o.status, o.currency, o.paid_amount_minor, o.placed_at, "
            "o.delivered_at, array_agg(i.product_name ORDER BY i.id) AS product_names "
            f"FROM {SCHEMA}.orders o JOIN {SCHEMA}.order_items i ON i.order_id = o.id "
            "WHERE o.customer_id = :customer "
            "GROUP BY o.id ORDER BY o.placed_at DESC, o.id LIMIT :limit"
        ),
        {"customer": customer.customer_id, "limit": limit},
    ).all()
    return [OrderSummary.model_validate(row._mapping) for row in rows]
