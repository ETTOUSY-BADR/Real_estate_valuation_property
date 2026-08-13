# Model artifacts

Training workflows write serialized estimators beneath this directory. Model
files are intentionally excluded from Git because they are large generated
artifacts. Rebuild them with the phase workflow scripts from the repository
root.

The Phase 3 integrity tests run automatically when the expected selected-model
artifact is available and skip with a clear message in a fresh clone.
