"""
This module defines the following routines used by the 'transform' step:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""

def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import make_column_transformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    import numpy as np
    
    # get the list of columns names of categorical variables
    cat_selector = make_column_selector(dtype_include=object)
    # get the list of column names of numerical variables
    num_selector = make_column_selector(dtype_include=np.number)


    cat_linear_processor = OneHotEncoder(handle_unknown="ignore")

    num_linear_processor = make_pipeline(
        StandardScaler(), SimpleImputer(strategy="mean", add_indicator=True)
    )

    linear_preprocessor = make_column_transformer(
        (num_linear_processor, num_selector), (cat_linear_processor, cat_selector)
    )

    return linear_preprocessor
    #
    # FIXME::OPTIONAL: return a scikit-learn-compatible transformer object.
    #
    # Identity feature transformation is applied when None is returned.
    return None
