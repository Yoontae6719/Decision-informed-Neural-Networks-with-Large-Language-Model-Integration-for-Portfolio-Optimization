import itertools
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
from nltk.corpus import wordnet

import nltk
from nltk.corpus import wordnet

def synonym_replacer(sentence):
    words = sentence.split()
    new_sentence = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            # 원래 단어를 제외한 동의어 리스트 생성
            synonym_words = [lemma.name().replace('_', ' ') for syn in synonyms for lemma in syn.lemmas()]
            synonym_words = list(set(synonym_words))  # 중복 제거
            synonym_words = [w for w in synonym_words if w.lower() != word.lower()]  # 원래 단어 제외
            if synonym_words:
                synonym = random.choice(synonym_words)
                new_sentence.append(synonym)
            else:
                new_sentence.append(word)
        else:
            new_sentence.append(word)
    return ' '.join(new_sentence)

def generate_market_prompt(pred_len, global_market_view, use_synonym=False):
    # 사용자 전망이 없는 경우 처리
    if not global_market_view or global_market_view == [None]:
        market_prompt_ = "<|start_prompt|>\n"
        market_prompt_ += "The user did not provide any market outlook.\n"
        market_prompt_ += "<|end_prompt|>"
        return market_prompt_
    
    # 기존 로직 유지
    # 선택된 지표 이름 추출
    indicator_names = [view[0] for view in global_market_view if view is not None]
    
    # 템플릿 기반으로 프롬프트 생성
    market_prompt_ = f"<|start_prompt|>Dataset purpose description: This dataset includes key economic indicators such as {', '.join(indicator_names)}. These indicators help forecast stock returns and guide robust portfolio optimization. In general, ICSA, UNRATE, and HYBS are more positive as they fall, while UMCSENT and HSN1F are more positive as they rise.\n\n"
    market_prompt_ += f"Task description: The user provides their outlook for the next 12 months on a monthly basis. These forecasts are used to guide decision-making and portfolio optimization. Note that the model predicts over the next {pred_len} steps, which may differ from the user's outlook period.\n\n"
    market_prompt_ += f"Objective: We need to create an optimal portfolio based on the user's forecast for economic indicators over the next 12 months.\n\n"
    market_prompt_ += "User overview:\n"
    
    # 월 이름 리스트 생성
    month_names = ["September", "October", "November", "December", "January", "February", "March", "April", "May", "June",
                   "July", "August"]

    # 사용자 제공 전망 추가
    for view in global_market_view:
        if view is None:
            continue
        indicator = view[0]
        confidence = view[1]
        forecasts = view[2:]
        provided_forecasts = [(i, forecast) for i, forecast in enumerate(forecasts) if forecast is not None]
        if provided_forecasts:
            market_prompt_ += f"- For {indicator}, the user provides {confidence.lower()} forecasts for the following months:\n"
            for idx, forecast in provided_forecasts:
                month = month_names[idx % 12]
                market_prompt_ += f"  {month}: {forecast}\n"
        else:
            market_prompt_ += f"- For {indicator}, the user did not provide any forecasts.\n"
    
    market_prompt_ += "<|end_prompt|>"

    if use_synonym:
        market_prompt_ = synonym_replacer(market_prompt_)

    return market_prompt_

def generate_stocks_prompt(pred_len, global_stock_view, ticker_dict, sector_dict, use_synonym=False):
    # 티커 사전을 역으로 매핑하여 티커에서 인덱스로 접근할 수 있게 함
    ticker_dict_inv = {v: k for k, v in ticker_dict.items()}
    
    # 사용자 전망이 없는 경우 처리
    if not global_stock_view or global_stock_view == [None]:
        stocks_prompt_ = "<|start_prompt|>\n"
        stocks_prompt_ += "The user did not provide any stock outlook.\n"
        stocks_prompt_ += "<|end_prompt|>"
        return stocks_prompt_
    
    # 기존 로직 유지
    stocks_prompt_ = "<|start_prompt|>\n"
    stocks_prompt_ += "Dataset purpose description: This dataset includes stock performance data used for forecasting and portfolio optimization.\n"
    stocks_prompt_ += "Each column represents a specific stock, and the column index corresponds to the stock as per the provided ticker dictionary. Sector information for each stock is also included.\n\n"
    stocks_prompt_ += "Task description: The user provides their outlook on stock performance, which includes comparisons between stocks over the next year and individual stock forecasts on a quarterly basis.\n\n"
    stocks_prompt_ += "Objective: We need to create an optimal portfolio based on the user's forecasts.\n\n"
    stocks_prompt_ += "User overview:\n"
    
    for view in global_stock_view:
        if view is None:
            continue
        if len(view) == 4:
            if view[2] in ["Highly confident", "Confident", "Moderately confident", "Uncertain"]:
                # 주식 간 비교 처리
                stock_1, stock_2, confidence, direction = view
                index_1 = ticker_dict_inv.get(stock_1)
                index_2 = ticker_dict_inv.get(stock_2)
                sector_1 = sector_dict.get(stock_1, "Unknown Sector")
                sector_2 = sector_dict.get(stock_2, "Unknown Sector")
                if index_1 is not None and index_2 is not None:
                    stocks_prompt_ += f"- The user is {confidence.lower()} that {stock_1} (column {index_1}, {sector_1}) will {direction} compared to {stock_2} (column {index_2}, {sector_2}) over the next year.\n"
                else:
                    stocks_prompt_ += f"- The user is {confidence.lower()} that {stock_1} ({sector_1}) will {direction} compared to {stock_2} ({sector_2}) over the next year.\n"
            elif view[2] in ["First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter"]:
                # 개별 주식의 분기별 전망 처리
                stock, confidence, quarter, forecast = view
                index = ticker_dict_inv.get(stock)
                sector = sector_dict.get(stock, "Unknown Sector")
                if index is not None:
                    stocks_prompt_ += f"- The user is {confidence.lower()} about {stock}'s (column {index}, {sector}) performance in the {quarter} with a forecast of {forecast}.\n"
                else:
                    stocks_prompt_ += f"- The user is {confidence.lower()} about {stock}'s ({sector}) performance in the {quarter} with a forecast of {forecast}.\n"
            else:
                # 퍼센티지 값이 주식 간 비교에 포함된 경우
                stock_1, stock_2, direction, percentage = view
                index_1 = ticker_dict_inv.get(stock_1)
                index_2 = ticker_dict_inv.get(stock_2)
                sector_1 = sector_dict.get(stock_1, "Unknown Sector")
                sector_2 = sector_dict.get(stock_2, "Unknown Sector")
                if index_1 is not None and index_2 is not None:
                    stocks_prompt_ += f"- The user expects {stock_1} (column {index_1}, {sector_1}) to {direction} by {percentage} compared to {stock_2} (column {index_2}, {sector_2}) over the next year.\n"
                else:
                    stocks_prompt_ += f"- The user expects {stock_1} ({sector_1}) to {direction} by {percentage} compared to {stock_2} ({sector_2}) over the next year.\n"
        elif len(view) == 5:
            # 주식 간 비교에서 퍼센티지와 자신감 수준 모두 제공된 경우
            stock_1, stock_2, confidence, direction, percentage = view
            index_1 = ticker_dict_inv.get(stock_1)
            index_2 = ticker_dict_inv.get(stock_2)
            sector_1 = sector_dict.get(stock_1, "Unknown Sector")
            sector_2 = sector_dict.get(stock_2, "Unknown Sector")
            if index_1 is not None and index_2 is not None:
                stocks_prompt_ += f"- The user is {confidence.lower()} that {stock_1} (column {index_1}, {sector_1}) will {direction} by {percentage} compared to {stock_2} (column {index_2}, {sector_2}) over the next year.\n"
            else:
                stocks_prompt_ += f"- The user is {confidence.lower()} that {stock_1} ({sector_1}) will {direction} by {percentage} compared to {stock_2} ({sector_2}) over the next year.\n"
        else:
            # 예외 처리
            stocks_prompt_ += f"- The user provided an unrecognized format: {view}\n"
    
    stocks_prompt_ += "<|end_prompt|>"
    
    if use_synonym:
        stocks_prompt_ = synonym_replacer(stocks_prompt_)
    
    return stocks_prompt_



def user_view_(stock_term, pred_len, llm_model, tokenizer, ticker_dict, sector_dict, global_market_views=None, global_stocks_views=None, use_synonym=False):
    # Batch 크기 가져오기
    B = stock_term.size(0)
    market_prompt = []
    stocks_prompt = []

    for b in range(1):
        # 시장 전망에서 무작위 부분집합 선택

        # 시장 전망 프롬프트 생성
        market_prompt_ = generate_market_prompt(pred_len, global_market_views, use_synonym)
        market_prompt.append(market_prompt_)


        # 주식 전망 프롬프트 생성
        stocks_prompt_ = generate_stocks_prompt(pred_len, global_stocks_views,ticker_dict,sector_dict, use_synonym)
        stocks_prompt.append(stocks_prompt_)

    # 프롬프트를 임베딩으로 변환
    market_prompt_tokenized = tokenizer(market_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
    stocks_prompt_tokenized = tokenizer(stocks_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=4598)

    market_prompt_embeddings = llm_model.get_input_embeddings()(market_prompt_tokenized.input_ids.to(stock_term.device))
    stocks_prompt_embeddings = llm_model.get_input_embeddings()(stocks_prompt_tokenized.input_ids.to(stock_term.device))

    return market_prompt_embeddings, stocks_prompt_embeddings, market_prompt, stocks_prompt





##################
import random  # Make sure to import random if not already imported

def invert_confidence(confidence): 
    confidence_levels = ["Highly confident", "Confident", "Moderately confident", "Uncertain"]
    inverted_levels = confidence_levels[::-1]
    confidence_map = dict(zip(confidence_levels, inverted_levels))
    return confidence_map.get(confidence, "Uncertain")

def invert_direction(direction):
    direction_map = {
        "outperform": "underperform",
        "underperform": "outperform",
        "increase": "decrease",
        "decrease": "increase",
        "rise": "fall",
        "fall": "rise"
    }
    return direction_map.get(direction.lower(), "change")

def generate_negative_market_prompt(pred_len, global_market_view, use_synonym=False):
    if not global_market_view or global_market_view == [None]:
        market_prompt_ = "<|start_prompt|>\n"
        market_prompt_ += "The user did not provide any market outlook.\n"
        market_prompt_ += "<|end_prompt|>"
        return market_prompt_
    
    # Extract indicator names
    indicator_names = [view[0] for view in global_market_view if view is not None]
    
    market_prompt_ = f"<|start_prompt|>Dataset purpose description: This dataset includes key economic indicators such as {', '.join(indicator_names)}. These indicators help forecast stock returns and guide robust portfolio optimization. In general, ICSA, UNRATE, and HYBS are more positive as they fall, while UMCSENT and HSN1F are more positive as they rise.\n\n"
    
    market_prompt_ += f"Task description: The user provides their outlook for the next 12 months on a monthly basis. These forecasts are used to guide decision-making and portfolio optimization. Note that the model predicts over the next {pred_len} steps, which may differ from the user's outlook period.\n\n"
    
    market_prompt_ += f"Objective: We need to create an optimal portfolio based on the user's forecast for economic indicators over the next 12 months.\n\n"
    
    market_prompt_ += "User overview:\n"
    
    month_names = ["September", "October", "November", "December", "January", "February", "March", "April", "May", "June",
                   "July", "August"]

    # Reverse user's forecasts
    for view in global_market_view:
        if view is None:
            continue
        indicator = view[0]
        confidence = invert_confidence(view[1])
        forecasts = view[2:]
        provided_forecasts = [(i, forecast) for i, forecast in enumerate(forecasts) if forecast is not None]
        if provided_forecasts:
            market_prompt_ += f"- For {indicator}, the user provides {confidence.lower()} forecasts for the following months:\n"
            for idx, forecast in provided_forecasts:
                month = month_names[idx % 12]
                # Use the forecast value as is
                market_prompt_ += f"  {month}: {forecast}\n"
        else:
            market_prompt_ += f"- For {indicator}, the user did not provide any forecasts.\n"
        
    market_prompt_ += "<|end_prompt|>"

    if use_synonym:
        market_prompt_ = synonym_replacer(market_prompt_)

    return market_prompt_

def generate_negative_stocks_prompt(pred_len, global_stock_view, ticker_dict, sector_dict, use_synonym=False):
    ticker_dict_inv = {v: k for k, v in ticker_dict.items()}
    
    if not global_stock_view or global_stock_view == [None]:
        stocks_prompt_ = "<|start_prompt|>\n"
        stocks_prompt_ += "The user did not provide any stock outlook.\n"
        stocks_prompt_ += "<|end_prompt|>"
        return stocks_prompt_
    
    stocks_prompt_ = "<|start_prompt|>\n"
    stocks_prompt_ += "Dataset purpose description: This dataset includes stock performance data used for forecasting and portfolio optimization.\n"
    stocks_prompt_ += "Each column represents a specific stock, and the column index corresponds to the stock as per the provided ticker dictionary. Sector information for each stock is also included.\n\n"
    stocks_prompt_ += "Task description: The user provides their outlook on stock performance, which includes comparisons between stocks over the next year and individual stock forecasts on a quarterly basis.\n\n"
    stocks_prompt_ += "Objective: We need to create an optimal portfolio based on the user's forecasts.\n\n"
    stocks_prompt_ += "User overview:\n"
    
    for view in global_stock_view:
        if view is None:
            continue
        if len(view) == 4:
            if view[2] in ["Highly confident", "Confident", "Moderately confident", "Uncertain"]:
                # Stock comparison with confidence
                stock_1, stock_2, confidence, direction = view
                inverted_confidence = invert_confidence(confidence)
                inverted_direction = invert_direction(direction)
                index_1 = ticker_dict_inv.get(stock_1)
                index_2 = ticker_dict_inv.get(stock_2)
                sector_1 = sector_dict.get(stock_1, "Unknown Sector")
                sector_2 = sector_dict.get(stock_2, "Unknown Sector")
                if index_1 is not None and index_2 is not None:
                    stocks_prompt_ += f"- The user is {inverted_confidence.lower()} that {stock_1} (column {index_1}, {sector_1}) will {inverted_direction} compared to {stock_2} (column {index_2}, {sector_2}) over the next year.\n"
                else:
                    stocks_prompt_ += f"- The user is {inverted_confidence.lower()} that {stock_1} ({sector_1}) will {inverted_direction} compared to {stock_2} ({sector_2}) over the next year.\n"
            elif view[2] in ["First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter"]:
                # Individual stock quarterly forecasts
                stock, confidence, quarter, forecast = view
                inverted_confidence = invert_confidence(confidence)
                index = ticker_dict_inv.get(stock)
                sector = sector_dict.get(stock, "Unknown Sector")
                if index is not None:
                    stocks_prompt_ += f"- The user is {inverted_confidence.lower()} about {stock}'s (column {index}, {sector}) performance in the {quarter} with a forecast of {forecast}.\n"
                else:
                    stocks_prompt_ += f"- The user is {inverted_confidence.lower()} about {stock}'s ({sector}) performance in the {quarter} with a forecast of {forecast}.\n"
            else:
                # Percentage value included in stock comparison
                stock_1, stock_2, direction, percentage = view
                percentage = percentage.strip('%')  # Remove '%' symbol
                try:
                    percentage = float(percentage)  # Convert to float
                except ValueError:
                    print(f"Invalid percentage value: {percentage}")
                    continue  # Skip this view or handle the error
                inverted_direction = invert_direction(direction)
                # Random noise addition
                noise = random.uniform(-0.05, 0.05)  # -5% to +5% random noise
                noisy_percentage = percentage + noise * abs(percentage)
                index_1 = ticker_dict_inv.get(stock_1)
                index_2 = ticker_dict_inv.get(stock_2)
                sector_1 = sector_dict.get(stock_1, "Unknown Sector")
                sector_2 = sector_dict.get(stock_2, "Unknown Sector")
                if index_1 is not None and index_2 is not None:
                    stocks_prompt_ += f"- The user expects {stock_1} (column {index_1}, {sector_1}) to {inverted_direction} by {noisy_percentage:.2f}% compared to {stock_2} (column {index_2}, {sector_2}) over the next year.\n"
                else:
                    stocks_prompt_ += f"- The user expects {stock_1} ({sector_1}) to {inverted_direction} by {noisy_percentage:.2f}% compared to {stock_2} ({sector_2}) over the next year.\n"
        elif len(view) == 5:
            # Stock comparison with percentage and confidence
            stock_1, stock_2, confidence, direction, percentage = view
            percentage = percentage.strip('%')  # Remove '%' symbol
            try:
                percentage = float(percentage)  # Convert to float
            except ValueError:
                print(f"Invalid percentage value: {percentage}")
                continue  # Skip this view or handle the error
            inverted_confidence = invert_confidence(confidence)
            inverted_direction = invert_direction(direction)
            # Random noise addition
            noise = random.uniform(-0.05, 0.05)
            noisy_percentage = percentage + noise * abs(percentage)
            index_1 = ticker_dict_inv.get(stock_1)
            index_2 = ticker_dict_inv.get(stock_2)
            sector_1 = sector_dict.get(stock_1, "Unknown Sector")
            sector_2 = sector_dict.get(stock_2, "Unknown Sector")
            if index_1 is not None and index_2 is not None:
                stocks_prompt_ += f"- The user is {inverted_confidence.lower()} that {stock_1} (column {index_1}, {sector_1}) will {inverted_direction} by {noisy_percentage:.2f}% compared to {stock_2} (column {index_2}, {sector_2}) over the next year.\n"
            else:
                stocks_prompt_ += f"- The user is {inverted_confidence.lower()} that {stock_1} ({sector_1}) will {inverted_direction} by {noisy_percentage:.2f}% compared to {stock_2} ({sector_2}) over the next year.\n"
        else:
            # Exception handling
            stocks_prompt_ += f"- The user provided an unrecognized format: {view}\n"
    
    stocks_prompt_ += "<|end_prompt|>"
    
    if use_synonym:
        stocks_prompt_ = synonym_replacer(stocks_prompt_)
    
    return stocks_prompt_

def user_negative_view_(stock_term, pred_len, llm_model, tokenizer, ticker_dict, sector_dict, global_market_views=None, global_stocks_views=None, use_synonym=False):
    # Get batch size
    B = stock_term.size(0)
    market_prompt = []
    stocks_prompt = []

    for b in range(1):
        # Generate market prompt (negative outlook)
        market_prompt_ = generate_negative_market_prompt(pred_len, global_market_views, use_synonym)
        market_prompt.append(market_prompt_)

        # Generate stocks prompt (negative outlook)
        stocks_prompt_ = generate_negative_stocks_prompt(pred_len, global_stocks_views, ticker_dict, sector_dict, use_synonym)
        stocks_prompt.append(stocks_prompt_)

    # Tokenize prompts
    market_prompt_tokenized = tokenizer(market_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
    stocks_prompt_tokenized = tokenizer(stocks_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=4598)

    # Convert token IDs to embeddings
    market_prompt_embeddings = llm_model.get_input_embeddings()(market_prompt_tokenized.input_ids.to(stock_term.device))
    stocks_prompt_embeddings = llm_model.get_input_embeddings()(stocks_prompt_tokenized.input_ids.to(stock_term.device))

    return market_prompt_embeddings, stocks_prompt_embeddings, market_prompt, stocks_prompt
