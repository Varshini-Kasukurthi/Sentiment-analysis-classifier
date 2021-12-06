
from app import Sentiment
import dataset_path

dataset = dataset_path.dataset_path()

App = Sentiment(dataset)

# load dataset and return data read
data = App.load_dataset()

# print(data)
# print("\n")
# print("\n")


# Clean text
data['text'] = [App.clean_text(text) for text in data['text']]
# print(data['text'])
# print("\n")
# print("\n")


# # data splitting
x_train_data, y_train_data = App.data_split(data)
# print("This is x train data!")
# print(x_train_data)

# print("\n")
# print("\n")

# print("This is y train data!")
# print(y_train_data)

# print("\n")
# print("\n")

# # We have to vectorize our data, train and test data to get a csr matrix
x_train_data_vector, y_train_data_labels, x_test_data_vector, y_test_data_labels = App.vectorize_text(x_train_data, y_train_data)

# print("This is x train data vector!")
# print(x_train_data_vector)
# print("\n")
# print("\n")

# print("This is x train data labels vector!")
# print(y_train_data_labels)
# print("\n")
# print("\n")

# print("This is x test data labels vector!")
# print(x_test_data_vector)
# print("\n")
# print("\n")

# print("This is x test data vector!")
# print(y_test_data_labels)
# print("\n")
# print("\n")

# # We have to dome feature scaling and 
# # have the scale matrix from our data vectors
x_train_data_scale, y_test_data_scale = App.scale_feature_matrix(x_train_data_vector, x_test_data_vector)
# print("This is x train data vector scale matrix!")
print(x_train_data_scale)
print("\n")
print("\n")

print("This is y test data vector scale matrix!")
print(y_test_data_scale)
print("\n")
print("\n")


# # Do some training, by parsing a .fit() method from the svm Classifier
# #Do some predictions and return the predictions
model, predictions = App.model(x_train_data_scale, y_train_data_labels, x_test_data_vector, y_test_data_labels)
print("This is the Y Predictions!")
print(predictions)
print("\n")
print("\n")

# # Apply error matrix, simply known as confusion matrix.
# # Error Matrix visualization
confusion_matrix_visualization = App.error_matrix(y_test_data_labels, predictions)
confusion_matrix_visualization.show()
