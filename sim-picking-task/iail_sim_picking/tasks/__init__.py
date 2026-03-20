from iail_sim_picking.tasks.packing_google_objects import PackingSeenGoogleObjectsSeq
from iail_sim_picking.tasks.packing_google_objects import PackingUnseenGoogleObjectsSeq
from iail_sim_picking.tasks.packing_google_objects import PackingExtraGoogleObjectsSeq

names = {
    "ur5f": PackingSeenGoogleObjectsSeq,
    "ur5l": PackingUnseenGoogleObjectsSeq,
    "ur5r": PackingExtraGoogleObjectsSeq,
}
