"""Shipment read service (W1.12): the shipping provider, after an ownership check."""

from sqlalchemy.orm import Session

from app.auth.customer import AuthenticatedCustomer
from app.schemas.shipping import ShipmentView
from app.services.errors import ServiceFailure
from app.services.orders import ensure_order_owner
from app.services.providers import ProviderClient, parse


def get_shipment_status(
    db: Session, shipping: ProviderClient, customer: AuthenticatedCustomer, order_id: str
) -> ShipmentView:
    # The provider is only asked about an order the customer owns.
    ensure_order_owner(db, customer, order_id)
    shipments = shipping.get("/shipments", params={"order_id": order_id})
    if not shipments:
        raise ServiceFailure("SHIPMENT_NOT_FOUND", "The courier has no shipment for this order.")
    # One shipment per order today; if there are more, the latest update is the one to report.
    latest = max(shipments, key=lambda s: s["last_updated_at"])
    return parse(ShipmentView, latest, "shipping")
