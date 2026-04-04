# DeskFit AI Dataset Documentation

## Overview

This dataset contains **100 curated entries** of micro-fitness exercises, posture correction tips, and wellness advice specifically designed for professionals working long hours at desks.

| File | Entries | Description |
|------|---------|-------------|
| `exercises.json` | 50 | Desk stretches, breathing exercises, eye care, hand/wrist exercises, micro-strength movements |
| `posture_tips.json` | 25 | Ergonomic desk setup, sitting posture, keyboard/mouse positioning, standing desk guidelines |
| `wellness_advice.json` | 25 | Nutrition, sleep, stress management, mental health, recovery strategies |

## Schema Documentation

### exercises.json

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (format: `ex_XXX`) |
| `title` | string | Exercise name |
| `category` | string | One of: `stretch`, `strengthening`, `breathing`, `eye_care`, `hand_wrist` |
| `body_area` | string | One of: `neck`, `shoulders`, `back`, `wrists`, `eyes`, `legs`, `full_body` |
| `duration_seconds` | int | Estimated time to complete |
| `difficulty` | string | One of: `beginner`, `intermediate` |
| `requires_equipment` | bool | Whether any equipment is needed |
| `can_do_at_desk` | bool | Whether it can be performed at a desk |
| `description` | string | Detailed description of the exercise |
| `steps` | string[] | Step-by-step instructions |
| `benefits` | string[] | Health benefits |
| `best_for` | string[] | Scenarios when this exercise is most useful |
| `precautions` | string[] | Safety warnings and contraindications |
| `source` | string | Evidence source or authority |

### posture_tips.json

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (format: `pt_XXX`) |
| `title` | string | Tip name |
| `category` | string | One of: `ergonomics`, `sitting_posture`, `typing_posture` |
| `applies_to` | string | Context: `desk_setup`, `sitting_position`, `hand_position`, `phone_use`, `commute`, `environment`, `work_habits` |
| `description` | string | Detailed explanation |
| `quick_fix` | string | Immediate actionable advice |
| `signs_of_problem` | string[] | Symptoms that indicate this tip is relevant |
| `source` | string | Evidence source |

### wellness_advice.json

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (format: `wa_XXX`) |
| `title` | string | Advice title |
| `category` | string | One of: `eye_care`, `nutrition`, `recovery`, `sleep`, `stress_management`, `mental_health`, `productivity` |
| `context` | string | When this advice applies |
| `description` | string | Detailed explanation |
| `why_it_works` | string | Scientific rationale |
| `when_to_use` | string[] | Situations where this advice is most relevant |
| `source` | string | Evidence source |

## Annotation Guidelines

All entries follow these quality standards:
- **Evidence-based**: each entry cites a credible source (medical research, professional guidelines, or recognized experts)
- **Actionable**: every entry provides specific, immediately implementable advice
- **Desk-appropriate**: exercises and tips are designed for office environments
- **Safety-conscious**: precautions and contraindications are included where relevant
- **Diverse coverage**: entries span multiple body areas, difficulty levels, and wellness domains

## Dataset Statistics

- **Body areas covered**: neck, shoulders, back, wrists, eyes, legs, full body
- **Exercise categories**: 5 (stretch, strengthening, breathing, eye_care, hand_wrist)
- **Difficulty levels**: beginner (majority), intermediate
- **Duration range**: 20 seconds to 5 minutes
- **Equipment needed**: 49 of 50 exercises require no equipment
