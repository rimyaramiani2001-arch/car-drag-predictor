# Car Drag Predictor

Building AI course project

## Summary

An AI tool that predicts a car's aerodynamic drag coefficient (Cd) from a few basic body dimensions, such as length, width, height and windshield angle. Designers get a quick early estimate before running slow and costly simulations or wind tunnel tests.

## Background

Aerodynamic drag affects fuel consumption, electric range, top speed and noise. The drag coefficient (Cd) is one of the most important numbers in car design, and designers want to improve it from the very first sketches.

* Measuring Cd needs wind tunnel tests or computational fluid dynamics (CFD) simulations, which are slow and expensive.
* In the early concept phase, designers change proportions many times and cannot simulate every variation.
* A quick approximate estimate would help them compare shapes and discard poor ideas early.

My personal motivation is a strong interest in car design. I want to understand how much of a car's efficiency can already be predicted from simple proportions, using only the methods learned in this course.

## How is it used?

The tool is meant for use in the early concept phase of car design.

1. The designer enters basic dimensions of a planned car: length, width, height, windshield angle, rear slope, ground clearance.
2. The model returns an estimated Cd value.
3. The designer changes the numbers (for example, a lower roofline) and sees right away how the estimate changes.

Intended users are design students, hobbyist car designers and small design teams without access to CFD software. The estimate is a rough guide only. It does not replace real aerodynamic testing, and final decisions must be based on simulation or wind tunnel results. People affected by the results are mainly the designers themselves, and indirectly the future owners of the cars, who benefit from more efficient vehicles.

## Data sources and AI methods

**Data.** The model needs a table where each row is one car model, with its basic dimensions as inputs and the published Cd value as the output. Publicly available manufacturer specification sheets and open car datasets can be used. Only data that is allowed to be reused is included, and the sources are listed under Acknowledgments.

**AI methods.**

* **Linear regression** as the first, simple and explainable baseline. Each coefficient shows how much one dimension changes the predicted Cd, and the coefficients are estimated with least squares.
* **Nearest neighbor method** as a second approach. The Cd of a new design is predicted from the most similar known cars. Features should be scaled so that, for example, length in millimetres does not dominate angles in degrees.
* **Training and test data.** The data is split so that the model is always evaluated on cars it has not seen during training. Otherwise the results would look better than they really are.

Example of the idea in code. The numbers below are made-up placeholders only to show the structure, not real measurements:

```python
import numpy as np

# columns: length (m), width (m), height (m), windshield angle (deg)
x_train = np.array([
    [4.5, 1.8, 1.45, 60.0],
    [4.2, 1.7, 1.50, 58.0],
    [4.8, 1.9, 1.40, 63.0],
    [4.4, 1.8, 1.60, 55.0],
    [4.6, 1.8, 1.42, 62.0],
])
y_train = np.array([0.28, 0.31, 0.26, 0.34, 0.27])  # placeholder Cd values

# add an intercept column and estimate the coefficients with least squares
X = np.column_stack([np.ones(len(x_train)), x_train])
c = np.linalg.lstsq(X, y_train, rcond=None)[0]

new_design = np.array([1.0, 4.5, 1.8, 1.43, 61.0])
print("Estimated Cd:", new_design @ c)
```

## Challenges

* Cd depends on many details that basic dimensions do not capture: underbody shape, wheels and tires, mirrors, cooling openings, spoilers. The model cannot see any of these.
* The model is only as good as its data. If the data contains mostly one type of car, such as sedans, predictions for very different shapes (SUVs, sports cars) will be unreliable.
* With few training examples, a model with many inputs can overfit and look accurate on training data but fail on new designs.
* Published Cd values are measured with different methods and are not always directly comparable.
* The tool says nothing about safety, comfort, manufacturing or style. Aerodynamics is only one part of good design.

## What next?

* Collect a larger and more varied dataset, including electric vehicles.
* Add more features, for example frontal area, rear slope and ground clearance.
* Compare linear regression, nearest neighbor and other methods, and report the errors on test data.
* Show the uncertainty of each prediction instead of a single number.
* Use image-based methods (convolutional neural networks) to estimate Cd from side-view sketches or photos.
* Build a simple web page where designers can try different dimensions.

Help is needed from people with experience in vehicle aerodynamics and in collecting and cleaning car data.

## Acknowledgments

* The Elements of AI: Building AI course by Reaktor Innovations and the University of Helsinki, which provided the README template and the methods used here (linear regression, nearest neighbor, training and test data).
* Data sources: to be listed here with the original creators and licenses once the dataset is chosen. Only data that may legally be reused is included.
