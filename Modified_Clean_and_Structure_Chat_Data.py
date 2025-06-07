
# Import Required Libraries
import pandas as pd
import re

# Function to Read Data
def read_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to Clean Transcript: Remove emails, names, and timestamps
def clean_transcript(transcript):
    cleaned_transcript = re.sub(r'\S+@\S+', '', transcript)  # Remove emails
    cleaned_transcript = re.sub(r'\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC\)', '', cleaned_transcript)  # Remove timestamps
    cleaned_transcript = re.sub(r'^\w+:', '', cleaned_transcript)  # Remove names
    return cleaned_transcript.strip()

# Function to Extract and Classify Lines
def extract_and_classify_lines(transcript):
    lines = transcript.split('\n')
    structured_lines = []
    for line in lines:
        if line:
            # Check if the line contains a colon
            if ':' in line:
                speaker, line_text = line.split(':', 1)
            else:
                # Skip this line or use a default value for speaker and line_text
                continue  # or speaker, line_text = 'Unknown', line
                
            line_type = "Question" if "?" in line_text else "Statement"
            structured_lines.append({
                'Speaker': speaker.strip(),
                'Line_Type': line_type,
                'Line_Text': line_text.strip()
            })
    return structured_lines

# Main Function
if __name__ == "__main__":
    file_path = "/Users/iandeuberry/Downloads/RawDataClean.csv"  # Replace with the path to your chat log CSV file
    df = read_data(file_path)
    
    structured_data_list = []
    
    # Check if 'chat_id' exists in DataFrame columns
    if 'chat_id' in df.columns:
        conversation_id_column = 'chat_id'
    else:
        print("The column 'chat_id' does not exist. Please check the CSV file.")
        exit(1)
        
    for index, row in df.iterrows():
        transcript = row['transcript']
        
        cleaned_transcript = clean_transcript(transcript)
        
        structured_lines = extract_and_classify_lines(cleaned_transcript)
        
        for line_data in structured_lines:
            line_data['Conversation_ID'] = row[conversation_id_column]
            structured_data_list.append(line_data)
    
    structured_data_df = pd.DataFrame(structured_data_list)
    
    structured_data_df.to_csv("Structured_Chat_Data.csv", index=False)  # This will save the structured data to a new CSV file
