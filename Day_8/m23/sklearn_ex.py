from google.cloud import storage
import pickle

BUCKET_NAME = 'example_bucket_m18'
MODEL_FILE = 'model.pkl'

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
my_model = pickle.loads(blob.download_as_string())

def knn_classifier(request):
   """ will to stuff to your request """
   request_json = request.get_json()
   if request_json and 'input_data' in request_json:
         data = request_json['input_data']
         input_data = list(map(int, data.split(',')))
         prediction = my_model.predict([input_data])
         return f'Belongs to class: {prediction}'
   else:
         return 'No input data received'
