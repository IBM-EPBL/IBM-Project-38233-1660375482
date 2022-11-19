from ..app import predict_values

def test_predict_orders():
    """ Predict orders """

    input_features = [0.0, 0.0, 2.0, 3.0, 647, 56, 0.0]
    features_name = ['homepage_featured', 'emailer_for_promotion', 'op_area', 'cuisine', 'city_code', 'region_code', 'category']

    predicted_value = predict_values(features_name, input_features)

    assert 'predictions' in predicted_value