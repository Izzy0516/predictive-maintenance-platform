# Project Guide

This is the longer companion to the [README](README.md). It explains what I built, why I made the specific technical decisions I did, and what the results actually mean. If the README is the storefront, this is the workshop.

## What the project does

I built a small machine learning system that predicts the Remaining Useful Life (RUL) of a jet engine, in operational cycles, from its sensor readings. The trained model is served through a FastAPI endpoint, so a caller sends a JSON payload of sensor values and gets back a predicted RUL plus a maintenance recommendation.

The end product is:

```
POST /predict
{
  "features": {
    "s2": 642.15,
    "s3": 1591.82,
    "s4": 1408.70,
    ...
  }
}

→
{
  "predicted_rul": 34.2,
  "recommendation": "schedule_inspection"
}
```

## Why this problem, specifically

I'm studying Motorsport Engineering, and RUL prediction is the same technical problem that sits behind component life management in motorsport. Gearbox change rules in F1, ERS component lifetimes, power unit life budgets: these are all "how much life does this component have left before it fails" decisions, made under uncertainty, from sensor and telemetry data.

The same problem shape shows up across industrial predictive maintenance, aviation, medical device monitoring, and datacentre operations. Companies like McLaren Applied and Williams Racing Technology sell software that solves exactly this. IBM has a whole product line (Maximo) built around it.

The reason I chose it as a portfolio project isn't just the domain fit. It's that the technical shape of the task, sensor time series turned into a supervised regression problem, is a compact way to demonstrate the entire ML engineering cycle in a single project: data cleaning, feature engineering, model selection, honest evaluation, serving, testing, and continuous integration.

## The dataset

I used NASA's C-MAPSS FD001 dataset. 100 turbofan engines, each run to failure, with 21 sensor readings and 3 operating settings recorded per cycle. Roughly 20,631 rows in total.

Run-to-failure data is rare and valuable, because in the real world most components are replaced before they fail. Nobody actually observes the "end of life" label. C-MAPSS gets around this by being simulated: NASA ran a physics-based turbofan model until each virtual engine broke, which means we have a true RUL label for every point in the history.

FD001 is the simplest of the four sub-datasets (single operating condition, single fault mode). Starting there let me get the pipeline right before adding complexity. FD002 through FD004 add multiple operating conditions and would need condition-normalisation preprocessing, which is genuine follow-up work.

## The early decisions that shaped everything else

There are four decisions in the pipeline that determine what all the later results mean. If any of them is wrong, the results are wrong, and no amount of model tuning will recover.

### 1. Reconstructing the target

RUL isn't measured directly, it's reconstructed retrospectively:

```
RUL = max_cycle_for_engine - current_cycle
```

This only works because every engine in the training set was run to failure. If any of them had been stopped early, I wouldn't know their true final cycle and their labels would be wrong.

The general lesson here is that in industry, this labelling strategy doesn't survive contact with reality: most fleets replace components before they fail. Any production system built on top of RUL prediction has to answer "how do we generate labels when nothing actually fails in the field?" That's usually solved with survival analysis or partially-labelled techniques. It was out of scope for this project, but it's the honest limitation of the approach.

### 2. Clipping the RUL target at 125 cycles

Raw RUL is a bad target. Here's why.

At cycle 10 of an engine that eventually runs for 200 cycles, the raw RUL is 190. But the engine is nominally healthy at cycle 10: the sensors are steady, degradation hasn't started, and the sensor readings look basically identical to another engine at cycle 20 that also has 180 cycles left. Asking the model to distinguish those two states is asking it to solve an impossible sub-problem. The result is that model capacity gets wasted on the healthy region, and predictions in the degraded region (where I actually care about accuracy) get blurred.

The fix, which is now standard in the C-MAPSS literature (Heimes 2008), is to clip the target at 125:

```python
RUL_clipped = min(RUL, 125)
```

This encodes a piecewise-linear degradation assumption: engines are considered "healthy" (with a constant RUL of 125) until they have 125 cycles left, at which point RUL starts a linear countdown to zero.

The evidence supports the choice. Pearson correlations between the sensors and raw RUL, split at the 125 boundary:

| sensor | RUL > 125 (healthy) | RUL ≤ 125 (degrading) |
|---|---|---|
| s2  | −0.13 | −0.67 |
| s4  | −0.16 | −0.74 |
| s11 | −0.17 | −0.77 |
| s12 | +0.16 | +0.74 |

The relationships are 4 to 5 times stronger once degradation has begun. The healthy region carries almost no signal about how much life is left.

**Why I still keep the healthy-region rows in training even though their target is now identically 125.** The model has to see healthy states, because in deployment it will start by monitoring healthy engines. Filtering them out would also be a subtle form of leakage, because "keep only rows where true RUL > X" is a rule that uses the label to decide the training set, and I can't reproduce that rule at inference time.

### 3. Per-engine rolling features

Raw sensor readings are noisy. Per-engine rolling means and rolling standard deviations over a 5-cycle window smooth out the jitter and expose the underlying degradation trend. For each of the 14 informative sensors I compute:

* `s{n}_rolling_mean`: mean of the last 5 cycles for this engine
* `s{n}_rolling_std`: standard deviation of the last 5 cycles for this engine

Two implementation details are worth spelling out.

**Grouping by engine.** The rolling window must not cross engine boundaries. Cycle 200 of engine 1 must not contaminate cycle 1 of engine 2. `groupby("engine_id").rolling(...)` enforces this.

**`min_periods=1` for the early cycles.** With a window of 5, the first four cycles of every engine would have no full window and would return NaN. That's 400 rows across the whole dataset, all of them in the healthy region. Dropping them would under-represent the healthy regime, and worse, deployment would break because a live engine at cycle 1 has no rolling window either. `min_periods=1` returns a partial-window mean instead, which is exactly what production would see.

For rolling std at cycle 1, the variance of a single point is undefined, so I fill with 0.0. Meaning: "no observed variation yet."

### 4. Dropping the constant sensors

Ten of the 24 raw columns have zero or near-zero variance across all engines: `setting_1`, `setting_2`, `setting_3`, `s1`, `s5`, `s6`, `s10`, `s16`, `s18`, `s19`. They carry no information on FD001 (which has a single operating condition), so I dropped them. That leaves 14 informative sensors, which with the rolling means and stds gives 42 features.

## Splitting the data honestly

This is the single most important decision in the whole project, and it's the one that most tutorials get wrong.

The rows in this dataset are **not independent**. Cycle 87 and cycle 88 of engine 1 are almost the same moment physically: the turbine has spun a few more times, s4 has moved by 0.3°C, s11 by 0.02. If I use scikit-learn's default `train_test_split(..., random_state=42)`, it shuffles all the rows randomly and about 20% of every engine's rows land in the test set. Every test row has a near-identical copy in the training set. That's not measuring generalisation, it's measuring memorisation.

The consequence is a fake RMSE of around 12 to 14 on FD001, versus the honest 18 to 22 that the literature reports. Any reviewer who knows the dataset spots this immediately.

**The fix is to split by engine.** Some engines go entirely into training, others go entirely into testing. Scikit-learn's `GroupKFold`, given `groups=engine_id`, guarantees this. The generalisation being measured is now the one that matters operationally: how well does the model perform on engines it has never seen?

This is a specific case of a general principle: your split has to mirror the deployment gap. In production, the model will see users, or patients, or customers, or engines that it wasn't trained on. So the test set has to contain the same kind of unseen entity. For fraud detection the equivalent is the customer. For medical ML it's the patient. For C-MAPSS it's the engine.

I use 5-fold GroupKFold. On 100 engines, each fold holds out 20 for testing and trains on the other 80. Every engine is held out exactly once, and the reported score is the mean plus the spread across folds. The spread matters too: a stable model has similar RMSE across folds. Wild variation would suggest the model is fragile to which engines it happens to see.

## The metrics, and why I report two

### RMSE

Root mean squared error. Standard regression metric. Square root of the mean of the squared errors, expressed in the original units (cycles):

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

What RMSE bakes in:

* **Symmetric.** Being 20 cycles too early scores the same as being 20 cycles too late.
* **Quadratic in error.** Being off by 20 counts as 4 times worse than being off by 10.
* **In original units.** "RMSE of 18" reads naturally as "off by roughly 18 cycles on average."

What RMSE hides in this domain: late and early errors are not equally bad. Predicting "40 cycles left" when the truth is 10 (engine dies in service) is catastrophically worse than predicting "40 cycles left" when the truth is 70 (slightly premature maintenance). RMSE is blind to that asymmetry, and in a safety-critical setting that blindness is a real problem.

### The NASA C-MAPSS scoring function

Defined by the challenge organisers to encode the operational asymmetry that RMSE ignores. For each prediction, let `d = predicted - true`. Then:

$$s_i = \begin{cases} e^{-d_i / 13} - 1 & \text{if } d_i < 0 \quad \text{(early)} \\ e^{d_i / 10} - 1 & \text{if } d_i \geq 0 \quad \text{(late)} \end{cases}$$

Total score is the sum across all predictions. Lower is better. A perfect model scores 0.

The key properties:

* **Asymmetric.** Denominator 10 for late errors, 13 for early. A 30-cycle late error costs about 19.09, while a 30-cycle early error costs about 9.05. Same magnitude, roughly twice the penalty for being late.
* **Exponential.** Being 60 cycles late costs about 20 times what being 30 cycles late costs. Catastrophic misses dominate the score.
* **Summed, not averaged.** Bigger fleet, more chances to make a catastrophic mistake, higher possible score. Never compare NASA totals across differently-sized datasets without dividing by row count.

### Why both

Reporting only RMSE hides the failure mode that matters most in the real world. Reporting only NASA score gives no intuition about typical error size. Together they tell a complete story.

A worked example. Suppose you have two models:

* Model A: RMSE 18, NASA 340
* Model B: RMSE 16, NASA 620

Model B has lower RMSE but a much higher NASA score. That means Model B has smaller average errors but is more prone to *late* predictions on near-failure engines, which is the operationally dangerous failure mode. In a safety-critical deployment, Model A is the right choice. RMSE alone would have picked wrong.

## The baseline

Before any real model, I established a floor. The **mean predictor** ignores every feature and predicts the training-set mean for every input. It's the dumbest possible predictor that still respects the problem shape, and its RMSE is provably equal to the standard deviation of the target.

Any real model has to beat this to demonstrate it's actually learned something from the sensors.

On FD001, with 5-fold engine-grouped CV:

* Mean RMSE: 41.67 (which equals `std(y_clipped)`)
* Mean NASA score: about 1,271,000
* Fold-to-fold spread: 0.00

Two things are worth noting:

**The exact match between RMSE and std(y).** This is the mathematical identity, and seeing it hold on real data is a cross-check on the whole pipeline. If the metric, the splits, or the baseline were broken, the identity wouldn't hold.

**The gap between the two metrics.** The baseline's RMSE (41.67) sounds mediocre but not disastrous. Its NASA score (1.27 million) is catastrophic. The reason is that the mean predictor confidently outputs "86 cycles left" for engines that are actually about to fail, and a single row where the model predicts 86 when the true RUL is 5 contributes about 3,300 to the NASA total. RMSE averages that away. NASA refuses to.

This is precisely why the domain metric exists. The mean predictor isn't just numerically mediocre, it's operationally dangerous.

## The model comparison

I compared four models on the held-out `test_FD001.txt` (100 engines, one prediction per engine at the last recorded cycle):

| Model | RMSE | NASA score |
|---|---|---|
| Mean baseline | 41.94 | 33,354 |
| Ridge regression | 21.03 | 1,337 |
| Random Forest | 18.26 | 1,149 |
| **HistGradientBoosting** | **17.93** | **865** |

Gradient boosting's 17.93 RMSE is close to Saxena et al. (2008)'s benchmark of 18.4 on this exact file. That was the target: hit the published benchmark range with defensible methodology.

### The interesting finding

The rank ordering on the NASA score is different in cross-validation than on the held-out set.

In 5-fold CV on the training data, Ridge was the NASA winner. That's because CV evaluates on every cycle of every held-out engine, including many near-failure rows. Tree ensembles' averaging bias pulls their predictions toward the middle, which produces many small late predictions on near-failure rows, and the exponential NASA penalty punishes exactly that failure mode.

On the held-out test set, predictions are only made on the last recorded cycle of each engine, and most of those engines are still relatively healthy at the cutoff. Tree ensembles' averaging bias helps on healthy rows and hurts on near-failure ones. On this specific slice of the data, gradient boosting wins on both metrics.

The takeaway is that "best model" is a function of the operational regime. If a deployed system is going to spend most of its time monitoring healthy engines and occasionally alerting on near-failure ones, the trade-offs are different than in a system that only makes predictions right before failure.

## The API

The trained model is served through a FastAPI endpoint. Two endpoints:

* `GET /health`: liveness check. Returns 200 with a JSON payload indicating whether the model is loaded. This is the standard endpoint that any ops or monitoring system hits to verify the service is alive.
* `POST /predict`: takes a JSON body of sensor features, returns a predicted RUL and a maintenance recommendation.

Some engineering choices worth flagging:

**Startup handling.** The model is loaded once at application startup using FastAPI's `lifespan` context manager (the modern replacement for the deprecated `on_event` hook). Loading takes a moment, so doing it per request would waste thousands of milliseconds. Doing it at startup means the very first request is fast.

**Input validation.** Callers must supply all the features the model was trained on. Missing features return HTTP 400 with a helpful message listing what's missing, rather than a silent crash inside sklearn. Malformed request bodies return HTTP 422 from Pydantic's automatic validation.

**Feature ordering.** The training script persists the feature column list alongside the model, and the API reindexes incoming requests to that exact order before calling `.predict()`. Dictionaries have no guaranteed order, so without this step a request could silently map features to the wrong slots and produce a wrong answer with no error.

**Business logic layer.** The raw prediction (a floating-point number of cycles) isn't useful to a maintenance engineer on its own. The `_recommend()` function translates it into one of three categories: `healthy`, `schedule_inspection`, `immediate_inspection`. That translation is the thing that makes the API useful. It's a small demonstration that I understand a model isn't a product; a system built around a model is.

## Testing

39 tests, all passing. They encode design decisions rather than incidental behaviour. If someone (future me, a collaborator, an AI assistant) breaks any of these properties by accident, the test fires immediately.

* **Metrics.** Perfect prediction gives 0. RMSE is symmetric in sign. NASA is asymmetric (late > early). NASA aggregates by sum, not mean.
* **Data pipeline.** RUL is 0 at the last cycle of each engine. RUL clipping caps the healthy region. Rolling windows never bleed across engines. No NaN rows appear after feature engineering.
* **Splits.** No engine appears in both train and test of any fold. Every row appears in exactly one test fold. The splitter fails loudly on bad input.
* **Models.** The baseline's RMSE equals std(y). Ridge beats the baseline on data with linear signal. The Ridge pipeline includes the scaler step.
* **API.** `/health` returns 200. `/predict` returns the expected schema. Missing features return 400. Malformed body returns 422. The recommendation is always consistent with the predicted RUL.

## What I'd do next if I had more time

* **Hyperparameter tuning.** I used defaults for Ridge (alpha=1.0) and gradient boosting. `GridSearchCV` on those hyperparameters, with GroupKFold to preserve the split discipline, would likely give a small RMSE improvement.
* **Error analysis.** A residuals plot and an analysis of the five worst-predicted engines would probably reveal a pattern, most likely that short-lived engines with unusual degradation trajectories are harder to predict.
* **Extending to FD002-004.** These add multiple operating conditions, which need condition-normalisation before the same pipeline applies. Real work, but the natural next step.
* **A live deployment.** Right now the API runs locally. Deploying to a public URL would give a shareable link. Free-tier options have all recently added credit-card requirements, so I've deferred this.

## What I deliberately didn't build

* **A frontend.** The FastAPI `/docs` page is already an interactive interface.
* **A database.** The trained model is a file on disk. Adding Postgres to a stateless prediction service would be complexity without value.
* **State-of-the-art performance.** Beating the published benchmark takes deep learning and months of tuning. Matching it with clean engineering demonstrates the skills I wanted to show.

## One-sentence summary

An end-to-end ML system that predicts Remaining Useful Life on jet engine sensor data, using tree-based regressors evaluated with both symmetric RMSE and the domain-specific asymmetric NASA scoring function under engine-grouped cross-validation to avoid leakage, and exposed through a tested FastAPI endpoint with continuous integration.

## Reference

Heimes, F. O. (2008). *Recurrent neural networks for remaining useful life estimation.* International Conference on Prognostics and Health Management.

Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation.* PHM 2008.