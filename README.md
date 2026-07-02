# Coach Tree

An analysis of the NFL coaching tree that models coaching relationships as a genetic pedigree. Coaching
"genes" are standardized coach-season behavioral phenotypes (z-scored residuals of observed play-calling
against play-level expectation models), and the project studies how those strategic traits transmit
through mentor-protege ties and how they relate to coaching performance (Coach WAR).

## Preregistration

Confirmatory heritability and selection analyses are preregistered on the Open Science Framework:

- Registration: https://osf.io/y2kr5/ (registered 2026-07-01)
- DOI: https://doi.org/10.17605/OSF.IO/Y2KR5
- Protocol: `outputs/reports/EHS_preregistration.pdf` (source: `outputs/reports/EHS_preregistration.tex`)

The WAR phenotype is drawn from the sibling repository
https://github.com/jwilliamson7/Coach_WAR (pinned at commit `dffe2f1` in the preregistration).

## Repository layout

- `crawlers/` - Pro Football Reference scrapers and shared team/role constants
- `scripts/` - data processing, predictive models, gene analysis, and visualization
- `data/` - raw and processed datasets, including the coaching genes
- `models/` - trained XGBoost expectation models
- `outputs/` - analysis results, reports, and visualizations
- `utils/` - shared model pipeline, feature, and path helpers

See `CLAUDE.md` for a detailed architecture and command reference.
