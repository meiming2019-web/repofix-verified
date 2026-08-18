"""Reusable complete successful chain for final-evaluation tests."""

from dataclasses import dataclass

import pytest

from repofix.policy import PolicyVerificationResult, verify_patch_policy
from tests.unit.hidden.conftest import prepared_hidden
from tests.unit.policy.conftest import PreparedPolicy, prepared_policy
from tests.unit.regression.conftest import prepared_baseline, prepared_verification


_IMPORTED_FIXTURES = (
    prepared_baseline,
    prepared_verification,
    prepared_hidden,
    prepared_policy,
)


@dataclass
class PreparedFinalEvaluation:
    policy_chain: PreparedPolicy
    policy_result: PolicyVerificationResult

    def arguments(self) -> dict[str, object]:
        hidden_chain = self.policy_chain.hidden_chain
        public = hidden_chain.public
        return {
            "task": public.task,
            "reproduction_expectation": public.expectation,
            "regression_specification": public.specification,
            "hidden_specification": hidden_chain.specification,
            "policy_specification": self.policy_chain.specification,
            "original_reproduction_result": public.original,
            "proposal": public.proposal,
            "regression_baseline_result": public.baseline,
            "application_result": public.application,
            "post_patch_reproduction_result": public.post_patch,
            "regression_verification_result": hidden_chain.regression,
            "hidden_verification_result": self.policy_chain.hidden_result,
            "policy_verification_result": self.policy_result,
        }


@pytest.fixture
def prepared_final_evaluation(
    prepared_policy: PreparedPolicy,
) -> PreparedFinalEvaluation:
    hidden_chain = prepared_policy.hidden_chain
    public = hidden_chain.public
    policy_result = verify_patch_policy(
        workspace_root=public.workspace,
        task=public.task,
        policy_specification=prepared_policy.specification,
        proposal=public.proposal,
        application_result=public.application,
        post_patch_reproduction_result=public.post_patch,
        regression_verification_result=hidden_chain.regression,
        hidden_verification_result=prepared_policy.hidden_result,
    )
    return PreparedFinalEvaluation(
        policy_chain=prepared_policy,
        policy_result=policy_result,
    )
