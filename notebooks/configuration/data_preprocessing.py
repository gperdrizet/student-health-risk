# Conditions for the main preprocessing sweep.

conditions = {
    # One-hot encoding with KNN imputation (3, 5, 7 neighbors) - missing indicator
    'One-hot encoding, KNN imputation (3 neighbors), -indicator': {
        'Encoder': 'One-hot encoder',
        'Imputer': 'KNN imputer',
        'KNN neighbors': 3,
        'Missing indicator': False
    },
    'One-hot encoding, KNN imputation (5 neighbors), -indicator': {
        'Encoder': 'One-hot encoder',
        'Imputer': 'KNN imputer',
        'KNN neighbors': 5,
        'Missing indicator': False
    },
    'One-hot encoding, KNN imputation (7 neighbors), -indicator': {
        'Encoder': 'One-hot encoder',
        'Imputer': 'KNN imputer',
        'KNN neighbors': 7,
        'Missing indicator': False
    },

    # One-hot encoding with KNN imputation (3, 5, 7 neighbors) + missing indicator
    'One-hot encoding, KNN imputation (3 neighbors), +indicator': {
        'Encoder': 'One-hot encoder',
        'Imputer': 'KNN imputer',
        'KNN neighbors': 3,
        'Missing indicator': True
    },
    'One-hot encoding, KNN imputation (5 neighbors), +indicator': {
        'Encoder': 'One-hot encoder',
        'Imputer': 'KNN imputer',
        'KNN neighbors': 5,
        'Missing indicator': True
    },
    'One-hot encoding, KNN imputation (7 neighbors), +indicator': {
        'Encoder': 'One-hot encoder',
        'Imputer': 'KNN imputer',
        'KNN neighbors': 7,
        'Missing indicator': True
    },

    # Target encoding (smoothing 0.1) with KNN imputation (3, 5, 7 neighbors) - missing indicator
    'Target encoding (smoothing 0.1), KNN imputation (3 neighbors), -indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 0.1,
        'KNN neighbors': 3,
        'Missing indicator': False
    },
    'Target encoding (smoothing 0.1), KNN imputation (5 neighbors), -indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 0.1,
        'KNN neighbors': 5,
        'Missing indicator': False
    },
    'Target encoding (smoothing 0.1), KNN imputation (7 neighbors), -indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 0.1,
        'KNN neighbors': 7,
        'Missing indicator': False
    },

    # Target encoding (smoothing 1.0) with KNN imputation (3, 5, 7 neighbors) - missing indicator
    'Target encoding (smoothing 1.0), KNN imputation (3 neighbors), -indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 1.0,
        'KNN neighbors': 3,
        'Missing indicator': False
    },
    'Target encoding (smoothing 1.0), KNN imputation (5 neighbors), -indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 1.0,
        'KNN neighbors': 5,
        'Missing indicator': False
    },
    'Target encoding (smoothing 1.0), KNN imputation (7 neighbors), -indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 1.0,
        'KNN neighbors': 7,
        'Missing indicator': False
    },

    # Target encoding (smoothing 10.0) with KNN imputation (3, 5, 7 neighbors) - missing indicator
    'Target encoding (smoothing 10.0), KNN imputation (3 neighbors), -indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 10.0,
        'KNN neighbors': 3,
        'Missing indicator': False
    },
    'Target encoding (smoothing 10.0), KNN imputation (5 neighbors), -indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 10.0,
        'KNN neighbors': 5,
        'Missing indicator': False
    },
    'Target encoding (smoothing 10.0), KNN imputation (7 neighbors), -indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 10.0,
        'KNN neighbors': 7,
        'Missing indicator': False
    },

    # Target encoding (smoothing 0.1) with KNN imputation (3, 5, 7 neighbors) + missing indicator
    'Target encoding (smoothing 0.1), KNN imputation (3 neighbors), +indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 0.1,
        'KNN neighbors': 3,
        'Missing indicator': True
    },
    'Target encoding (smoothing 0.1), KNN imputation (5 neighbors), +indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 0.1,
        'KNN neighbors': 5,
        'Missing indicator': True
    },
    'Target encoding (smoothing 0.1), KNN imputation (7 neighbors), +indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 0.1,
        'KNN neighbors': 7,
        'Missing indicator': True
    },

    # Target encoding (smoothing 1.0) with KNN imputation (3, 5, 7 neighbors) + missing indicator
    'Target encoding (smoothing 1.0), KNN imputation (3 neighbors), +indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 1.0,
        'KNN neighbors': 3,
        'Missing indicator': True
    },
    'Target encoding (smoothing 1.0), KNN imputation (5 neighbors), +indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 1.0,
        'KNN neighbors': 5,
        'Missing indicator': True
    },
    'Target encoding (smoothing 1.0), KNN imputation (7 neighbors), +indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 1.0,
        'KNN neighbors': 7,
        'Missing indicator': True
    },

    # Target encoding (smoothing 10.0) with KNN imputation (3, 5, 7 neighbors) + missing indicator
    'Target encoding (smoothing 10.0), KNN imputation (3 neighbors), +indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 10.0,
        'KNN neighbors': 3,
        'Missing indicator': True
    },
    'Target encoding (smoothing 10.0), KNN imputation (5 neighbors), +indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 10.0,
        'KNN neighbors': 5,
        'Missing indicator': True
    },
    'Target encoding (smoothing 10.0), KNN imputation (7 neighbors), +indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'KNN imputer',
        'Target smoothing': 10.0,
        'KNN neighbors': 7,
        'Missing indicator': True
    },

    # Target encoding (smoothing 0.1, 1.0, 10.0) with iterative imputation - missing indicator
    'Target encoding (smoothing 0.1), iterative imputation, -indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'Iterative imputer',
        'Target smoothing': 0.1,
        'Missing indicator': False
    },
    'Target encoding (smoothing 1.0), iterative imputation, -indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'Iterative imputer',
        'Target smoothing': 1.0,
        'Missing indicator': False
    },
    'Target encoding (smoothing 10.0), iterative imputation, -indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'Iterative imputer',
        'Target smoothing': 10.0,
        'Missing indicator': False
    },

    # Target encoding (smoothing 0.1, 1.0, 10.0) with iterative imputation + missing indicator
    'Target encoding (smoothing 0.1), iterative imputation, +indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'Iterative imputer',
        'Target smoothing': 0.1,
        'Missing indicator': True
    },
    'Target encoding (smoothing 1.0), iterative imputation, +indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'Iterative imputer',
        'Target smoothing': 1.0,
        'Missing indicator': True
    },
    'Target encoding (smoothing 10.0), iterative imputation, +indicator': {
        'Encoder': 'Target encoder',
        'Imputer': 'Iterative imputer',
        'Target smoothing': 10.0,
        'Missing indicator': True
    },

    # One-hot encoding with iterative imputation - missing indicator
    'One-hot encoding, iterative imputation, -indicator': {
        'Encoder': 'One-hot encoder',
        'Imputer': 'Iterative imputer',
        'Missing indicator': False
    },

    # One-hot encoding with iterative imputation + missing indicator
    'One-hot encoding, iterative imputation, +indicator': {
        'Encoder': 'One-hot encoder',
        'Imputer': 'Iterative imputer',
        'Missing indicator': True
    }
}
