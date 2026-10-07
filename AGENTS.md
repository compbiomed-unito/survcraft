## Coding
Code should be readable and, when possibile, concise. In particular, some validation of input is good but there is no need for overly long chunks of validation code that will hide the functional code.

## Documentation

### API documentation
API documentation should use consistent NumPy-style docstrings for public classes, functions, and methods. Include a short description and relevant Parameters, Returns, Attributes, Raises, and Warns sections; add Notes, Examples, and See Also where useful. Omit irrelevant or empty sections.
Document dataclass constructor parameters in class docstrings, including inherited options, effective defaults, and meaningful constraints. Document learned or initialized public attributes and when they become available.
Prioritize SurvivalPredictor, SurvivalSimulator, and the public survival and loss modules. Mathematical documentation should define notation and describe the implemented parameterization, censoring treatment, reductions, and relevant approximations or limitations.
Establish shared contracts in SurvivalEstimator, BaseInputAdapter, BaseSurvivalAdapter, BaseSurvivalModule, and BaseSurvivalLoss. Cover feature and target formats, event-indicator semantics, tensor shapes and dtypes, time handling, prediction modes and return types, device handling, module construction, fitted state, scalar loss outputs, and supported arithmetic loss composition.
Concrete classes should reference these contracts and document their specific behavior, overrides, and limitations without unnecessary duplication. Check documentation against the implementation and keep examples small and runnable.