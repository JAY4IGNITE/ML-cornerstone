# Testing Checklist

## Data Tests
- Schema validation
- Missing target detection
- Invalid type detection
- Missing-value report
- Duplicate detection
- Leakage checks
- Join cardinality checks

## ML Tests
- Reproducible split
- Pipeline fit and transform
- Inference on one row
- Inference on a batch
- Probability bounds
- Unknown category handling
- Missing value handling
- Artifact loading
- Feature-name consistency

## API Tests
- Health endpoint
- Model info endpoint
- Valid prediction
- Invalid payload
- Missing required field
- Invalid numerical value
- Unknown category
- Error response schema

## Frontend Tests
- Form validation
- Loading state
- Error state
- Result rendering
- Responsive layout
- Accessibility basics

## Execution Rule
Run tests after each major phase.
Never report tests as passed unless they were actually executed.
Include the command and real result in the final report.
