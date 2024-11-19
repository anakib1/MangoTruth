# Training guidelines

1. Check out examples at `detectors/scripts` directory.
2. Check out useful utilities at `detector/utils/training.py`.
3. Use `IDetector` interface for seamless integration of model to ecosystem.
2. **Always** Use `detectors/metrics` for uploading and reporting results. This ensures consistency between all runs and
   contributors.
3. Don't train on test set.
4. Use `mango-truth` datasets for format consistency. In case you need new dataset - upload it to `mango-truth`.
5. After adding new model don't forget to add it to readme.