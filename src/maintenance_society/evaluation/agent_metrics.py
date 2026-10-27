"""Commitments, independent peer recognition, and the final six rounds."""

from .metrics import summarise


def agent_metrics(workshop, condition, discussions, recognition):
    if not workshop.done:
        return None
    summary = summarise(workshop, "local_llm").to_dict()
    summary.pop("policy")
    by_round = {record.round_number: dict(zip(workshop.config.agent_ids, record.actions))
                for record in workshop.history if record.operational}
    declarations = fulfilled = 0
    for discussion in discussions:
        for entry in discussion.entries:
            if entry.proposal.intended_action == "maintain":
                declarations += 1
                fulfilled += by_round[discussion.round_number][entry.agent_id] == "maintain"
    communication = condition in "CD"
    summary["commitments"] = {
        "applicable": communication,
        "maintenance_declarations": declarations if communication else None,
        "fulfilled_maintenance_declarations": fulfilled if communication else None,
        "fulfilment_rate": fulfilled / declarations if communication and declarations else None,
    }
    last = workshop.history[-6:]
    ids = workshop.config.agent_ids
    summary["final_six_rounds"] = {
        "first_round": last[0].round_number, "last_round": last[-1].round_number,
        "group_points": sum(sum(record.points_awarded) for record in last),
        "maintenance_turns": sum(record.maintainers for record in last),
        "failure_events": sum(record.failure_event for record in last),
        "lost_rounds": sum(not record.operational for record in last),
        "wasted_wear_reduction": sum(record.wasted_wear_reduction for record in last),
        "maintenance_counts": {identity: sum(record.operational and record.actions[index] == "maintain"
                                             for record in last) for index, identity in enumerate(ids)},
    }
    complete = set(recognition) == set(ids)
    received = {identity: sum(credits.get(identity, 0) for credits in recognition.values()) for identity in ids}
    summary["recognition"] = {
        "complete": complete, "evaluators_completed": len(recognition),
        "allocation_matrix": recognition,
        # Primary totals require all six evaluators. Partial allocations remain in the matrix.
        "received_totals": received if complete else None,
        "received_shares": {identity: received[identity] / 600 for identity in ids} if complete else None,
    }
    return summary
