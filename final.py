import os
import json
import requests
import pprint
import time
import streamlit as st

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from datetime import datetime, timedelta
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory



load_dotenv() # load your .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")


def retrieve_from_endpoint(url: str) -> dict:
    headers = {"Authorization": SECTORS_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    return json.dumps(data)


@tool
def get_company_report(stock: str, sections: str = 'overview') -> str:
    """
    Get company report sections such as dividend, financials, future, management, overview, ownership, peers, and valuation from company or ticker such as BBRI, BBCA, BMRI, BYAN, etc.
    If no specific sections are mentioned, use "overview" as the default.

    :param stock: The stock symbol or ticker for the company (e.g., BBRI, BBCA, BMRI, BYAN).
    :param sections: The specific report sections to retrieve (dividend, financials, future, management, overview, ownership, peers, and valuation). Defaults to 'overview'.
    :return: A string response from the endpoint with the requested company report sections.
    """
    stock = stock.upper()
    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections={sections}"

    return retrieve_from_endpoint(url)


@tool
def get_top_companies_by_tx_volume(start_date: str, end_date: str, top_n: int = 1) -> str:
    """
    Get the list of most traded stocks top companies based on by transaction volume and price for a specified interval (up to 90 days).
    If no specific start and end dates are provided, use the current date (current date = {datetime.today()}) as both start and end.
    If no specific `top_n` value is provided, default to 1. The maximum allowed value for `top_n` is 10.

    :param start_date: The start date for the interval in 'YYYY-MM-DD' format. Defaults to the current date, which is {datetime.today()}.
    :param end_date: The end date for the interval in 'YYYY-MM-DD' format. Defaults to the current date, which is {datetime.today()}.
    :param top_n: The number of top companies to return. Defaults to 1, with a maximum value of 10.
    :return: A string response with the most traded companies based on volume or price in the specified interval. Mention if the date is a weekend and if the date has been adjusted.
    """

    # Convert string date to datetime object
    def is_weekend(date_str: str) -> bool:
        """Check if a given date is a weekend (Saturday or Sunday)."""
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.weekday() >= 5  # Saturday = 5, Sunday = 6

    def next_weekday(date_str: str) -> str:
        """Return the next weekday (Monday to Friday) if the given date is a weekend."""
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        while date_obj.weekday() >= 5:  # Skip weekends
            date_obj += timedelta(days=1)
        return date_obj.strftime('%Y-%m-%d')
    
    change = 0

    # Check if start_date or end_date falls on a weekend and adjust to the next weekday
    if is_weekend(start_date):
        adjusted_start_date = next_weekday(start_date)
        print(f"Start date {start_date} is on a weekend, adjusted to {adjusted_start_date}.")
        start_date = adjusted_start_date
        change = 1
    
    if is_weekend(end_date):
        adjusted_end_date = next_weekday(end_date)
        print(f"End date {end_date} is on a weekend, adjusted to {adjusted_end_date}.")
        end_date = adjusted_end_date
        change = 1

    url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}"

    # return retrieve_from_endpoint(url)
    response_data = retrieve_from_endpoint(url)

    if isinstance(response_data, dict) and 'error' in response_data:
        return json.dumps(response_data)
    
    data_dict = json.loads(response_data)
    aggregated_data = {}
    for date_data in data_dict.values():
        for company in date_data:
            symbol = company['symbol']
            volume = company['volume']
            if symbol in aggregated_data:
                aggregated_data[symbol]['volume'] += volume
            else: 
                aggregated_data[symbol] = company.copy()
    
    data_sorted = sorted(aggregated_data.values(), key=lambda x: x['volume'], reverse=True)[:top_n]
    res = json.dumps(data_sorted)

    if change == 1:
        return [f"You can't answer date on weekends, tell you're sorry. Now the data shows for {start_date} - {end_date}. Show them this data is:", res]
    else:
        return [f"Show them this data is:", res]


@tool
def get_daily_tx(stock: str, start_date: str, end_date: str) -> str:
    """
    Get the daily transaction volume, close price, and market cap for a stock based on start date and end date provided..
    The available operations are: accumulation, average, min, and max. Defaults to accumulation.
    If the date falls on a weekend, it will be adjusted to the next available weekday, and a note will be included.
    
    :param stock: The stock symbol or ticker.
    :param start_date: The start date in 'YYYY-MM-DD' format.
    :param end_date: The end date in 'YYYY-MM-DD' format.
    :return: The result based on the transaction volume, close price, and market cap along with the notes. Mention if the date is a weekend and if the date has been adjusted.
    """
    stock = stock.upper()
    

    # Convert string date to datetime object
    def is_weekend(date_str: str) -> bool:
        """Check if a given date is a weekend (Saturday or Sunday)."""
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.weekday() >= 5  # Saturday = 5, Sunday = 6

    def next_weekday(date_str: str) -> str:
        """Return the next weekday (Monday to Friday) if the given date is a weekend."""
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        while date_obj.weekday() >= 5:  # Skip weekends
            date_obj += timedelta(days=1)
        return date_obj.strftime('%Y-%m-%d')
    
    change = 0

    # Check if start_date or end_date falls on a weekend and adjust to the next weekday
    if is_weekend(start_date):
        adjusted_start_date = next_weekday(start_date)
        print(f"Start date {start_date} is on a weekend, adjusted to {adjusted_start_date}.")
        start_date = adjusted_start_date
        change = 1
    
    if is_weekend(end_date):
        adjusted_end_date = next_weekday(end_date)
        print(f"End date {end_date} is on a weekend, adjusted to {adjusted_end_date}.")
        end_date = adjusted_end_date
        change = 1

    url = f"https://api.sectors.app/v1/daily/{stock}/?start={start_date}&end={end_date}"

    res = retrieve_from_endpoint(url)
    if change == 1:
        return [f"You can't answer date on weekends, tell you're sorry. Now the data shows for {start_date} - {end_date}. Show them this data is:", res]
    else:
        return [f"Show them this data is:", res]


@tool
def performance_since_ipo(stock: str) -> str:
    """
    Get performance since IPO listing for a given stock (e.g., BBRI, BBCA, BMRI, BYAN).
    Returns the percentage gain since the IPO listing date.
    Always evaluate the comparison based on the actual numerical values, including positive and negative signs
    
    :param stock: The stock symbol or ticker.
    :return: The percentage performance of the stock since its IPO. Sorted by 7 days, 30 days, 90 days, 365 days. Always evaluate the comparison based on the actual numerical values, including positive and negative signs
    """
    stock = stock.upper()
    url = f"https://api.sectors.app/v1/listing-performance/{stock}/"

    return retrieve_from_endpoint(url)


@tool
def Subsector_aggregated_stats(sub_sector: str = 'banks', sections: str = 'growth') -> str:
    """
    Get detailed statistics for given sectors organized into distinct sections (statistics, market_cap, stability, valuation, growth, companies). Default to 'statistics'.
    
    :param sub_sector: The sub sector name (e.g., 'alternative energy', 'banks', 'food beverage', 'insurance'). Defaults to 'banks'.
    :param sections: The specific report section ('companies', 'growth', 'market cap', 'stability', 'statistics', 'valuation'). Defaults to 'statistics'.
    :return: The result based on given sectors and sections.
    """
    sub_sector = sub_sector.strip().lower()
    sections = sections.strip().lower()
    url = f"https://api.sectors.app/v1/subsector/report/{sub_sector}/?sections={sections}"

    return retrieve_from_endpoint(url)



tools = [
    get_company_report,
    get_top_companies_by_tx_volume,
    get_daily_tx,
    performance_since_ipo,
    Subsector_aggregated_stats
]



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            You are a financial assistant of Indonesian stocks that provides precise and analytical responses using provided tools with high caliber capability on interpret and analyze the data to take perfect conclusions about it. 
            Always ensure responses are structured, accurate, informative, and relevant to stock market queries. Only provide data or answers if the query specifically pertains to Indonesian stocks.
            For queries about the other country of Indonesian such as Singapore, India, Japan, Korea, etc do not answer that queries and say sorry because you only provide data or answers if the query specifically pertains to Indonesian stocks..
            If the query does not relate to Indonesian stocks, respond with a message indicating that the service only provides data for Indonesian stocks and no information is available for the queried topic.
            
            General Guidelines:
            - For every response, include detailed explanations, data, or reasoning to support the answer.
            - If the query is not related to stocks, politely inform the user that you only handle stock-related queries.
            
            Date Handling:
            - When responding to date-related queries, assume today's date is {datetime.today()} unless specified otherwise.
            - For single-day queries, use the same date as both the start and end dates.
            - If the specified date falls on a weekend (Saturday or Sunday), mention that data is unavailable, and retrieve data for the next available business day.
            - When start and end dates are needed but not provided, infer these from the query (e.g., "last week," "past month," "three days ago").
            - Interpret ordinal terms like "first," "second," etc., as referring to specific days or rankings, and adjust start and end dates accordingly
            
            Tool Usage:
            - sections about dividend, financials, future, management, overview, ownership, peers, and valuation refer to get_company_report.
            - sections about statistics, market_cap, stability, valuation, growth, companies refer to Subsector_aggregated_stats.
            - Use the get_daily_tx tool for queries about daily transaction volume, close price, and market cap for a specific date range.
            - Use the get_top_companies_by_tx_volume tool for queries about the most traded stocks (by volume or price) over a specified period.
            - If you use get_company_report tool but couldn't define the sections params, set sections params default as 'overview'.
            - For queries about market cap refer to 'overview' in get_company_report tool.
            - For trend-based questions, provide insights into the most notable or relevant trends in the queried timeframe.
            
            Answering Multiple Questions:
            If there is more than one question, follow this steps:
            1. provide the first answer of the first question using relevant tool refering to tool usage.
            2. use the exact first answer to the relevant tool refering to the tool usage to get the next answer.

            - For queries that involve comparing data (e.g., higher, lower, smaller, bigger, etc.), always evaluate the comparison based on the actual numerical values, including positive and negative signs. Ensure that the comparison considers both the magnitude and direction (whether the values are positive or negative) to provide an accurate answer.
            - If a tool is invoked more than five times in a single response, stop invoking immediately and politely inform the user that further requests would be inefficient. Provide a summary based on the existing information instead. Don't be stupid.
            - Always ensure to use relevant tools when answering questions.
            - Present the answer using a markdown table if it is possible.
            
            """
        ),
        ("human", "{input}"),
        # msg containing previous agent tool invocations 
        # and corresponding tool outputs
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

llm = ChatGroq(
    temperature=0,
    model_name="llama3-groq-70b-8192-tool-use-preview",
    groq_api_key=GROQ_API_KEY,
)

agent = create_tool_calling_agent(llm, tools, prompt)
memoryforchat = ConversationBufferMemory()
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


st.title("💰 Stocks GPT")

#Sets up the session state to handle a chat or messaging interface
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I assist you today?"}]

#Make the content appears as chat text
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hi there, please say something about stocks"):
    st.chat_message("user").write(prompt) #Display the user's message in the chat
    st.session_state.messages.append({"role": "user", "content": prompt}) #Append the user's message to session state
    response = agent_executor.invoke({"input": prompt})

# Simulate thinking with a spinner while getting the agent's response
    with st.chat_message("assistant"): 
        with st.spinner("🧠 thinking..."):
            time.sleep(1) 
            try:
                response = agent_executor.invoke({"input": prompt})  #Invoke the agent to get the response based on the user's input
            except IndexError as e:
                st.error(f"Index Error: {e}")
                response = "Sorry there, something went wrong with the list handling."
            except Exception as e:
                st.error(f"Error invoking agent: {e}")
                response = "Sorry there, something went wrong."
            st.markdown(response['output'])
    st.session_state.messages.append({"role": "assistant", "content": response['output']})



