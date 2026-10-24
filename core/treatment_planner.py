"""Aggressive treatment-plan generator.

Non-discriminatory toward research chemicals per user request. Non-AGA
diagnoses route to disease-appropriate pathways.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

from core.alopecia_classifier import AlopeciaDiagnosis
from core.feature_extractor import TrichoscopicProfile


@dataclass
class TreatmentComponent:
    category: str
    agent: str
    dose_or_protocol: str
    evidence_tier: str
    rationale: str
    risk_notes: str


@dataclass
class TreatmentPlan:
    diagnosis_summary: str
    components: List[TreatmentComponent]
    monitoring: List[str]
    escalation_logic: List[str]
    full_disclosure: str

    def to_dict(self):
        return asdict(self)


AGGRESSIVE_DISCLOSURE = (
    "AGGRESSIVE PROTOCOL DISCLOSURE: This plan includes agents that are not approved for AGA "
    "(dutasteride, oral minoxidil) and at least one research chemical (RU58841) with limited human "
    "safety data. RU58841 is sold as a grey-market compound; purity and sterility are not guaranteed. "
    "Combining multiple anti-androgens increases the risk of systemic androgen suppression, sexual "
    "dysfunction, mood changes, and hepatic/cardiovascular effects. This output is an engineering "
    "artefact generated from trichoscopic features and published/community protocols; it is not a "
    "prescription and must be reviewed by a physician before any agent is used."
)


class AggressiveTreatmentPlanner:
    def plan(self, diagnosis: AlopeciaDiagnosis, profile: TrichoscopicProfile) -> TreatmentPlan:
        if diagnosis.primary == "AA":
            return self._aa_plan(diagnosis)
        if diagnosis.primary == "TE":
            return self._te_plan(diagnosis)
        if diagnosis.primary == "Scarring":
            return self._scarring_plan(diagnosis)
        return self._aga_plan(diagnosis, profile)

    def _aga_plan(self, d, p):
        components = [
            TreatmentComponent(
                category="Systemic",
                agent="Dutasteride",
                dose_or_protocol="0.5 mg orally once daily (or 0.5 mg 2-3x/week if tolerability is a concern).",
                evidence_tier="Regulatory (for BPH) / off-label clinical for AGA",
                rationale="Dual inhibition of 5AR type I and II; more complete scalp DHT suppression than finasteride.",
                risk_notes="Sexual dysfunction, depression, gynecomastia; long half-life means slow washout. Baseline PSA should be discussed.",
            ),
            TreatmentComponent(
                category="Topical",
                agent="RU58841 (PSK-3841)",
                dose_or_protocol="5% solution (~50 mg/mL), 1 mL nightly on dry scalp in a volatile delivery vehicle.",
                evidence_tier="Community / experimental (research chemical)",
                rationale="Non-steroidal androgen-receptor antagonist that blocks DHT at the follicle without systemic 5AR suppression. A 1% formulation produced a 103% increase in healthy follicle counts in human scalp xenografts.",
                risk_notes="Not FDA-approved; limited human safety data. Grey-market purity is variable. Rapid plasma half-life (~1 h) means nightly dosing is critical.",
            ),
            TreatmentComponent(
                category="Topical",
                agent="Pyrilutamide (KX-826)",
                dose_or_protocol="1.0% tincture twice daily to the target area.",
                evidence_tier="Phase III clinical trial (investigational access)",
                rationale="Topical AR inhibitor that reached its primary endpoint in a Chinese phase III AGA trial; 1.0% BID increased target-area non-vellus hair count by ~15.33 hairs/cm2 from baseline.",
                risk_notes="Still investigational; access is limited. Local irritation possible.",
            ),
            TreatmentComponent(
                category="Systemic",
                agent="Low-dose oral minoxidil",
                dose_or_protocol="2.5 mg orally once daily (range 1.25-5 mg/day for men).",
                evidence_tier="Off-label clinical (consensus-supported)",
                rationale="Bypasses topical adherence issues and stimulates anagen prolongation.",
                risk_notes="Fluid retention, tachycardia, hypertrichosis; baseline BP and renal function should be reviewed.",
            ),
            TreatmentComponent(
                category="Procedural",
                agent="Microneedling",
                dose_or_protocol="1.0 +/- 0.5 mm pen depth, once weekly (or every 2 weeks if erythema persists).",
                evidence_tier="Clinical evidence / off-label procedural",
                rationale="Micro-wounds upregulate growth factors and enhance topical penetration.",
                risk_notes="Pain, transient erythema, infection risk; do not share devices.",
            ),
            TreatmentComponent(
                category="Procedural",
                agent="Dutasteride mesotherapy",
                dose_or_protocol="0.1% dutasteride solution, 1 mL intradermally every 1-3 months (monthly induction x5, then quarterly maintenance).",
                evidence_tier="Off-label clinical / trial-supported",
                rationale="Delivers high local 5AR inhibition while limiting systemic exposure.",
                risk_notes="Injection-site pain, bruising, rare paradoxical shedding; must be performed by a trained clinician.",
            ),
            TreatmentComponent(
                category="Topical",
                agent="Ketoconazole 2% shampoo",
                dose_or_protocol="Apply to scalp 2-3x/week; leave 3-5 minutes before rinsing.",
                evidence_tier="Regulatory (antifungal) / off-label AGA adjunct",
                rationale="Reduces scalp inflammation and has weak local anti-androgen effect.",
                risk_notes="Scalp dryness; alternate with a gentle shampoo.",
            ),
            TreatmentComponent(
                category="Research chemical",
                agent="Topical finasteride 0.25% (compounded)",
                dose_or_protocol="1 mL to the scalp once daily - optional bridge if oral dutasteride is not tolerated.",
                evidence_tier="Off-label clinical / compounded",
                rationale="Topical finasteride 0.25% monotherapy has shown sustained improvement in hair density at 52 weeks with low systemic exposure.",
                risk_notes="Compounding pharmacy quality varies; systemic absorption is low but not zero.",
            ),
        ]
        monitoring = [
            "Baseline and 3-monthly: FBC, LFTs, renal function, PSA (if dutasteride).",
            "Blood pressure and heart rate at baseline and after oral minoxidil initiation.",
            "Trichoscopic re-imaging at 3, 6, and 12 months at the same calibrated scalp sites.",
            "Mood and sexual-function screening at each visit (5AR inhibitors).",
            "If RU58841 is used: document source and purity; consider baseline and 6-monthly lipids and LFTs.",
        ]
        escalation = [
            "If trichoscopic density continues to fall at 6 months despite oral dutasteride + topical RU58841 + oral minoxidil -> add dutasteride mesotherapy monthly.",
            "If diameter diversity improves but vellus ratio remains >20% at 12 months -> add pyrilutamide 1% BID or switch RU58841 vehicle.",
            "If side effects limit oral dutasteride -> switch to topical finasteride 0.25% + RU58841 while maintaining oral minoxidil.",
            "If the diagnosis shifts toward TE (empty-follicle ratio rises) -> investigate triggers before further AGA escalation.",
        ]
        summary = (f"Primary: AGA (confidence {d.confidence}). "
                   f"AGA severity hint: {d.aga_severity_hint}. "
                   f"Trichoscopic rationale: {'; '.join(d.rationale)}")
        return TreatmentPlan(summary, components, monitoring, escalation, AGGRESSIVE_DISCLOSURE)

    @staticmethod
    def _aa_plan(d):
        return TreatmentPlan(
            diagnosis_summary=f"Primary diagnosis: Alopecia Areata ({d.confidence} confidence). Rationale: {'; '.join(d.rationale)}.",
            components=[
                TreatmentComponent("Topical", "Clobetasol propionate 0.05% cream/foam",
                                   "Once daily to affected patches for 4-12 weeks, with pulsed breaks.",
                                   "Regulatory / clinical",
                                   "First-line localized immunosuppression for patchy AA.",
                                   "Skin atrophy with prolonged use."),
                TreatmentComponent("Procedural", "Intralesional triamcinolone acetonide",
                                   "2.5-10 mg/mL, 0.1 mL per site every 4-6 weeks.",
                                   "Clinical standard",
                                   "Targeted anti-inflammatory for patchy AA.",
                                   "Pain, dermal atrophy, hypopigmentation."),
                TreatmentComponent("Systemic", "Oral minoxidil 2.5 mg daily",
                                   "Adjunct for regrowth; does not treat the autoimmune mechanism.",
                                   "Off-label clinical",
                                   "Supports anagen; used alongside immunomodulation.",
                                   "Fluid retention, tachycardia, hypertrichosis."),
            ],
            monitoring=["Patch size and regrowth every 4-6 weeks.",
                        "Screen for other autoimmune conditions if widespread."],
            escalation_logic=["If patches expand or fail to respond -> dermatology referral for JAK-inhibitor evaluation."],
            full_disclosure=("AA is immune-mediated; AGA-style 5AR blockade is not the primary treatment. "
                             "This plan intentionally excludes research chemicals because the disease mechanism is different."),
        )

    @staticmethod
    def _te_plan(d):
        return TreatmentPlan(
            diagnosis_summary=f"Primary diagnosis: Telogen Effluvium ({d.confidence} confidence). Rationale: {'; '.join(d.rationale)}.",
            components=[
                TreatmentComponent("Systemic", "Trigger correction",
                                   "Iron/ferritin, thyroid, vitamin D, zinc, caloric intake, stress, medication review.",
                                   "Clinical standard",
                                   "TE is reactive; removing the trigger is definitive.",
                                   "Treat according to identified deficiency."),
                TreatmentComponent("Topical", "Minoxidil 5% topical (optional bridge)",
                                   "1 mL twice daily if shedding is severe and AGA co-existence is suspected.",
                                   "Regulatory",
                                   "May shorten the shedding phase and stimulate regrowth.",
                                   "Contact dermatitis, initial shed."),
            ],
            monitoring=["Repeat trichoscopy at 3 months; empty-follicle ratio should fall.",
                        "If diversity >20% emerges -> reassess for unmasking of AGA."],
            escalation_logic=["If shedding persists >6 months despite trigger correction -> biopsy or specialist referral."],
            full_disclosure=("TE does not warrant RU58841 or dutasteride mesotherapy unless AGA is also present. "
                             "This plan intentionally avoids unnecessary anti-androgen exposure."),
        )

    @staticmethod
    def _scarring_plan(d):
        return TreatmentPlan(
            diagnosis_summary=f"Primary diagnosis: Scarring alopecia suspected ({d.confidence} confidence). Rationale: {'; '.join(d.rationale)}.",
            components=[
                TreatmentComponent("Procedural", "Scalp biopsy (4 mm punch)",
                                   "For histopathology and direct immunofluorescence.",
                                   "Diagnostic standard",
                                   "Scarring alopecias require tissue diagnosis; trichoscopy alone is insufficient.",
                                   "Small scar; minimal risk."),
                TreatmentComponent("Systemic", "Specialist anti-inflammatory therapy (per histology)",
                                   "e.g., hydroxychloroquine, doxycycline, or immunosuppression depending on subtype.",
                                   "Specialist-directed",
                                   "Goal is to halt follicular destruction, not stimulate regrowth.",
                                   "Dependent on the agent."),
            ],
            monitoring=["Urgent dermatology referral.",
                        "Do not apply irritant research chemicals to inflamed scalp."],
            escalation_logic=["If biopsy confirms scarring -> manage the specific subtype (LPP, FFA, etc.); "
                              "hair transplantation is generally avoided in active disease."],
            full_disclosure=("Scarring alopecia is a destructive process. Aggressive AGA protocols are contraindicated "
                             "until the scarring component is controlled."),
        )
