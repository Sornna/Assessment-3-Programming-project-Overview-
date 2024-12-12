import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

class TESLAPricePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title('TESLA Price Prediction')

        # Load your dataset (replace 'TESLA.csv' with your dataset file)
        self.data = pd.read_csv('TESLA.csv')

        # Define predictors and target (adjust column names as per your dataset)
        self.predictors = ['Open', 'High', 'Low', 'Volume']
        self.target = 'Close'

        # Prepare data for training
        self.X = self.data[self.predictors].values
        self.y = self.data[self.target].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Train the model
        self.model = XGBRegressor()
        self.model.fit(self.X_train, self.y_train)

        # Initialize the GUI
        self.sliders = []
        self.create_widgets()

    def create_widgets(self):
        for i, column in enumerate(self.predictors):
            # Create a label for each feature
            label = tk.Label(self.master, text=f'{column}:')
            label.grid(row=i, column=0, padx=10, pady=5)

            # Create a current value display label
            current_val_label = tk.Label(self.master, text='0.0')
            current_val_label.grid(row=i, column=2, padx=10, pady=5)

            # Create a slider for each feature
            slider = ttk.Scale(
                self.master,
                from_=self.data[column].min(),
                to=self.data[column].max(),
                orient="horizontal",
                command=lambda val, label=current_val_label: label.config(text=f'{float(val):.2f}')
            )
            slider.grid(row=i, column=1, padx=10, pady=5)
            self.sliders.append(slider)

        # Create a Predict button
        predict_button = tk.Button(self.master, text='Predict Price', command=self.predict_price)
        predict_button.grid(row=len(self.predictors), columnspan=3, pady=10)

    def predict_price(self):
        # Collect input values from sliders
        inputs = [float(slider.get()) for slider in self.sliders]

        # Predict price using the model
        price = self.model.predict([inputs])

        # Display the prediction result
        messagebox.showinfo('Predicted Price', f'The predicted price is ${price[0]:.2f}')

if __name__ == '__main__':
    root = tk.Tk()
    app = TESLAPricePredictionApp(root)
    root.mainloop()
