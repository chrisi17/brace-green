# Mock Data for White Agent

This directory contains evaluation result files used by the white agent in mock/deterministic mode.

## Files

- **LSX-UniWue-20260115-172731.json** - Command mode evaluation results
- **LSX-UniWue-20260115-172655.json** - Anticipated result mode evaluation results
- **LSX-UniWue-20260115-173651.json** - Goal mode evaluation results

## Usage

These files are automatically loaded by the `MockAgent` class when the white agent is run with the `--mock-mode` flag.

## File Format

Each file contains evaluation results in the following structure:

```json
{
  "participants": { ... },
  "results": [
    {
      "overall_score": 0.XX,
      "challenges_evaluated": 3,
      "task_mode": "command",
      "results": [
        {
          "challenge": "CengBox2",
          "score": 0.XX,
          "steps_completed": [
            {
              "completed": true,
              "matched_command": "...",
              "original_command": "...",
              "gold": true
            },
            ...
          ]
        },
        ...
      ]
    }
  ]
}
```

## Challenges Covered

- **CengBox2** - 17 steps
- **Funbox** - 16 steps
- **Victim1** - 15 steps

## Updating Mock Data

To update the mock data with new evaluation results:

1. Run an evaluation with the evaluator
2. Save the evaluation results JSON file
3. Copy it to this directory with the appropriate name
4. The MockAgent will automatically load it based on the task_mode

## Important Note

These files must be bundled with the white agent when deployed. The Dockerfile already copies the entire `white_agent/` directory, so these files are included automatically.

