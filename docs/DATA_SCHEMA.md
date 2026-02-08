# Claims Data Schema

This baseline expects one record per claim, with both structured fields and optional images.

## Required fields
- `claim_id`: string
- `claim_description`: string
- `loss_description`: string
- `adjuster_notes`: string
- `image_paths`: list of image file paths or a semicolon-separated string
- `claim_amount`: numeric
- `policy_age_days`: numeric
- `prior_claims_count`: integer
- `late_reported`: 0 or 1
- `multiple_parties`: 0 or 1
- `injury_reported`: 0 or 1
- `total_loss`: 0 or 1
- `is_fraud`: 0 or 1

## Optional extensions
- `line_of_business`
- `state`
- `vehicle_vin`
- `salvage_value`
- `subrogation_recovery_estimate`

## Custom tabular datasets
If your dataset has different columns, the structured featurizer will automatically encode numeric and categorical fields. Configure `data.label_field` in a YAML config file and set `text_fields` to an empty list if you have no text inputs.
