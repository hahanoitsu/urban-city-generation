from __future__ import annotations

import torch
from torch import nn

from urban_model.city_planner import CityPlanner, CityPlannerConfig
from urban_model.plan_cell_graph import PlanCellGraphArchitect, PlanCellGraphConfig


CONTEXT_KEYS = (
    "context_cells",
    "context_line_points",
    "context_line_width",
    "context_line_length",
    "context_line_mode",
    "context_line_class",
    "context_line_vertical",
    "context_line_padding",
    "ports",
    "port_mode",
    "port_class",
    "port_vertical",
    "port_padding",
    "style",
    "controls",
)


def context_inputs(batch, use_controls=False, use_context=True):
    result = {key: batch[key] for key in CONTEXT_KEYS}
    if not use_controls:
        result["controls"] = torch.zeros_like(result["controls"])
    if not use_context:
        for key in CONTEXT_KEYS:
            if key not in ("style", "controls"):
                result[key] = torch.zeros_like(result[key])
        result["context_line_padding"] = torch.ones_like(result["context_line_padding"])
        result["port_padding"] = torch.ones_like(result["port_padding"])
    return result


class ContextPlanGraph(nn.Module):
    def __init__(self, planner_config, graph_config, normalization):
        super().__init__()
        self.planner = CityPlanner(CityPlannerConfig.from_dict(planner_config))
        self.graph = PlanCellGraphArchitect(PlanCellGraphConfig.from_dict(graph_config))
        for name in ("global_mean", "global_std", "presence_pos_weight"):
            self.register_buffer(name, torch.tensor(normalization[name], dtype=torch.float32))

    def forward(self, context, target):
        encoded = self.planner.context(context)
        plan = self.planner(context, encoded_context=encoded)
        graph = self.graph(target, context_memory=encoded[0])
        return plan, graph

    def predicted_plan(self, plan, *, stochastic=False, generator=None):
        # Undo the positive class weight before treating logits as probabilities.
        probability = torch.sigmoid(plan["plan_presence"].float() - self.presence_pos_weight.log())
        log_count = plan["plan_log_count"].float().clamp_max(8.0)
        count_mean = torch.expm1(log_count)
        counts = torch.round(count_mean * probability).clamp_min(0)
        if stochastic:
            occupied = torch.bernoulli(probability[..., 0], generator=generator)
            extra = torch.poisson((count_mean[..., 0] - 1).clamp_min(0), generator=generator)
            counts[..., 0] = occupied * (1 + extra)

        counts[..., 0].clamp_(max=self.graph.config.max_slots_per_cell)
        for row in counts:
            excess = int(row[:, 0].sum()) - self.graph.config.max_nodes
            if excess > 0:
                for index in torch.argsort(row[:, 0], descending=True):
                    removed = min(excess, int(row[index, 0]))
                    row[index, 0] -= removed
                    excess -= removed
                    if excess == 0:
                        break
        road_share = count_mean[..., 1] / count_mean[..., 1:3].sum(dim=-1).clamp_min(1e-6)
        counts[..., 1] = torch.round(counts[..., 0] * road_share)
        counts[..., 2] = counts[..., 0] - counts[..., 1]
        counts[..., 7] = counts[..., 7].minimum(counts[..., 0])
        raw = plan["plan_global"].float() * self.global_std + self.global_mean
        raw[:, 0] = counts[..., 0].sum(dim=1)
        maximum_edges = torch.minimum(
            raw[:, 0] * (raw[:, 0] - 1) / 2, torch.full_like(raw[:, 0], self.graph.config.max_edges)
        )
        raw[:, 1] = torch.round(raw[:, 1]).clamp_min(0).minimum(maximum_edges)
        raw[:, 2] = raw[:, 2].clamp_min(0).minimum(raw[:, 0])
        raw[:, 3:] = raw[:, 3:].clamp(0, 1)
        return {
            "plan_counts": counts,
            "plan_presence": (counts > 0).float(),
            "plan_log_counts": torch.log1p(counts),
            "plan_orientation": plan["plan_orientation"].float(),
            "plan_global": (raw - self.global_mean) / self.global_std,
            "plan_global_raw": raw,
            "node_count": raw[:, 0].long(),
        }

    @torch.inference_mode()
    def generate(self, context, *, stochastic=False, generator=None):
        context = {key: context[key] for key in CONTEXT_KEYS}
        encoded = self.planner.context(context)
        predicted = self.planner(context, encoded_context=encoded)
        plan = self.predicted_plan(predicted, stochastic=stochastic, generator=generator)
        graph = self.graph(plan, context_memory=encoded[0])
        return plan, graph
