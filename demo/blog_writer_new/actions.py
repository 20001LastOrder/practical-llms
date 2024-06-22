import json
import os
import re
import time
from typing import Any

import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings.base import Embeddings
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.text_splitter import (MarkdownTextSplitter,
                                     SentenceTransformersTokenTextSplitter)
from langchain.vectorstores.chroma import Chroma
from loguru import logger
from pydantic import ConfigDict

from sherpa_ai.actions import GoogleSearch
from sherpa_ai.actions.base import BaseAction


def get_file_path_input(instruction):
    while True:
        file_path = input(instruction)
        if os.path.exists(file_path):
            return file_path
        else:
            print("File path does not exist. Please try again.")


class ChunkDocument(BaseAction):
    name: str = "chunk_document"
    args: dict = []
    usage: str = "chunk_document"
    chunk_size: int = 3000
    chunk_overlap: int = 300

    def execute(self, dict_state: dict, **kwargs):
        if "transcript" in dict_state:
            raw_transcript = dict_state["transcript"]
        else:
            transcript_path = get_file_path_input(
                "Enter the path to the transcript file: "
            )
            with open(transcript_path, encoding="utf-8") as f:
                raw_transcript = f.read()

        markdown_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        transcript_chunks = markdown_splitter.create_documents([raw_transcript])
        return transcript_chunks


class GenerateInsight(BaseAction):
    name: str = "generate_insight"
    args: dict = []
    usage: str = "generate_insight"
    chat: Any = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo",
    )
    verbose: bool = False

    def execute(self, dict_state: dict, **kwargs):
        transcript_chunks = dict_state["chunk_document"]
        response = ""
        for i, text in enumerate(transcript_chunks):
            insights = self.transcript2insights(text.page_content)
            response = "\n".join([response, insights])
            if self.verbose:
                print(
                    f"\nInsights extracted from chunk {i+1}/{len(transcript_chunks)}:\n{insights}"
                )
        return response

    def transcript2insights(self, transcript):
        system_template = "You are a helpful assistant that summarizes transcripts of podcasts or lectures."
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_template = """From this chunk of a presentation transcript, extract a short list of key insights. \
            Each line of the transcript starts with the initials of the speaker, and each key insight has to preserve the attribution to speaker. \
            Skip explaining what you're doing, labeling the insights and writing conclusion paragraphs. \
            The insights have to be phrased as statements of facts with no references to the presentation or the transcript. \
            Statements have to be full sentences and in terms of words and phrases as close as possible to those used in the transcript. \
            Keep as much detail as possible. The transcript of the presentation is delimited in triple backticks.

            Desired output format:
            - [speaker initials]: [Key insight #1]
            - [speaker initials]: [Key insight #2]
            - [...]

            Transcript:
            ```{transcript}```"""
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

        result = self.chat(
            chat_prompt.format_prompt(transcript=transcript).to_messages()
        )

        return result.content


class GenerateOutline(BaseAction):
    name: str = "generate_outline"
    args: dict = []
    usage: str = "generate_outline"
    verbose: bool = True

    chat_4o: Any = ChatOpenAI(
        temperature=0,
        model="gpt-4o",
    )

    def execute(self, data):
        statements = data["generate_insight"]
        # Split the text into lines
        lines = statements.split("\n")

        # Initialize a dictionary to hold the processed lines
        processed_dict = {}

        # Initialize a line number counter
        line_number = 1

        # Iterate through each line
        for line in lines:
            # Skip lines that start with "Insights extracted from chunk"
            if line.startswith("Insights extracted from chunk"):
                continue
            # Skip empty lines
            if not line.strip():
                continue
            # Replace "- " at the beginning of the line with "- [line number] "
            processed_line = re.sub(r"^- ", "", line)  # Remove the leading "- "
            # Add the line to the dictionary with the line number as the key
            processed_dict[line_number] = processed_line
            # Increment the line number counter
            line_number += 1

            # Flatten the dictionary into a single string with the desired format
            processed_lines = [
                f"- [{line_number}] {statement}"
                for line_number, statement in processed_dict.items()
            ]
            processed_statements = "\n".join(processed_lines)

        system_template = """You are an experienced technical writer who is good at storytelling for technical topics."""
        system_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template = """organize the following insights in a logical order. \
                keep only numbers corresponding to the key insight and not the statements of the insights. \
                cluster points that talk about the same topic. \
                only include 3-5 key insights per topic. \
                combine each 3-5 topics into a higher level topic. \
                organize the information in a json structure.

            Insights:
            ```{insights}```"""
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

        outline = self.chat_4o(
            chat_prompt.format_prompt(insights=processed_statements).to_messages()
        )

        # Use regex to extract the JSON part
        json_match = re.search(r"```json(.*?)```", outline.content, re.DOTALL)

        # Check if the JSON part was found
        if json_match:
            json_str = json_match.group(1).strip()

            # Convert the JSON string to a JSON object
            outline_json = json.loads(json_str)

            # Print the JSON object
            print(json.dumps(outline_json, indent=2))
        else:
            print("No JSON part found in the outline text.")

        # Iterate over the JSON data to replace numbers with strings from p_dict
        for key, value in outline_json.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    outline_json[key][sub_key] = [
                        processed_dict[item] for item in sub_value
                    ]
            elif isinstance(value, list):
                outline_json[key] = [processed_dict[item] for item in value]

        if self.verbose:
            print(f"\nEssay outline: {outline.content}\n")
            print(f"\nEssay outline (text): {outline_json}\n")
        return outline_json


class NextOutline(BaseAction):
    name: str = "next_outline"
    args: dict = []
    usage: str = "next_outline"

    def execute(self, **kwargs):
        pass


class ReadOutlines(BaseAction):
    name: str = "read_outlines"
    args: dict = []
    usage: str = "read_outlines"

    def execute(self, **kwargs):
        pass


class Write(BaseAction):
    name: str = "write"
    args: dict = []
    usage: str = "write"

    def execute(self, **kwargs):
        pass


class WriteFile(BaseAction):
    name: str = "write_file"
    args: dict = []
    usage: str = "write_file"

    def execute(self, **kwargs):
        pass


def get_action_map():
    return {
        "chunk_document": ChunkDocument(),
        "generate_insight": GenerateInsight(),
        "generate_outline": GenerateOutline(),
        "google_search": GoogleSearch(),
        "next_outline": NextOutline(),
        "read_outlines": ReadOutlines(),
        "write": Write(),
        "write_file": WriteFile(),
    }
