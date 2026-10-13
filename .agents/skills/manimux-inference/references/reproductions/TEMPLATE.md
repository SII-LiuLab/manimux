# <Method> Reproduction Record

## 0. Status Snapshot

- Date:
- Owner:
- Current gate:
- Verified:
- Not verified:

## 1. Upstream Pin

- Paper:
- Official repository:
- Commit:
- Files inspected:
- Released checkpoint:

## 2. Official Contract

- Model/input/output:
- Action semantics:
- Horizon and `dt`:
- Runtime cadence:
- Default parameters:

## 3. Official Algorithm

Record formulas and their exact source-code order. Distinguish paper prose from executable code.

## 4. ManiMux Target Contract

Describe the policy, embodiment, wire format, canonical `ActionChunk`, Timeline and Executor path.

## 5. Fidelity Matrix

| Item | Official | ManiMux | Status | Evidence |
|---|---|---|---|---|

Use only: `exact`, `equivalent adaptation`, `different`, `not verified`.

## 6. Implementation Walkthrough

List each changed file, symbol, input/output shape and why the change is required.

## 7. Data and Statistics

Record source paths, generation command, schema, normalization formula and invalidation conditions.

## 8. Configuration

Explain every non-default parameter and the reason for its value.

## 9. Validation Ladder

### 9.1 Static/contract

### 9.2 Unit tests

### 9.3 Real model forward

### 9.4 Simulator

### 9.5 Real robot

Never promote evidence from one level to another.

## 10. Expected Evidence

List logs, metrics, recorder artifacts and pass/fail criteria.

## 11. Known Differences and Risks

## 12. Rollback

## 13. Reviewer Checklist
