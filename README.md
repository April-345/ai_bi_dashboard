# Conversational AI Business Intelligence Dashboard

An **AI-powered analytics dashboard** that allows users to ask questions about business data in **natural language** and automatically generates SQL queries, charts, and insights.

The system behaves like a simplified version of **Tableau, Power BI, or ThoughtSpot**, but powered by conversational AI.

Users can upload datasets and explore them without writing SQL.

---

## Features

* Natural language → SQL query generation
* Automatic data visualizations
* AI-generated insights
* CSV dataset upload
* Query history tracking
* Interactive dashboard UI
* Smart chart selection

---

## Example Queries

Try asking questions like:

* Average price by fuel type
* Top 10 most expensive cars
* Average mileage by transmission
* Price vs mileage
* Cheapest diesel cars under 20000

The system converts the question into SQL, executes the query, and visualizes the result.

---

## Tech Stack

**Frontend**

* Streamlit
* Plotly

**Backend**

* Python
* Pandas
* SQLite

**AI / NLP**

* Groq API
* Llama 3

---

## Project Structure

```
ai-bi-dashboard/
│
├── frontend/
│   └── app.py
│
├── backend/
│   ├── sql_generator.py
│   └── query_executor.py
│
├── data/
│   └── bmw.db
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/ai-bi-dashboard.git
cd ai-bi-dashboard
```

Install dependencies:

```
pip install -r requirements.txt
```

Set your Groq API key:

Windows:

```
set GROQ_API_KEY=your_api_key
```

Mac/Linux:

```
export GROQ_API_KEY=your_api_key
```

---

## Run the Application

Start the dashboard:

```
streamlit run frontend/app.py
```

Open your browser:

```
http://localhost:8501
```

---

## Future Improvements

* Multi-table database support
* Conversational query memory
* Advanced chart recommendations
* Cloud deployment

---

## License

MIT License
