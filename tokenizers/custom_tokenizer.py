# from __future__ import annotations
# from typing import Any, Dict, List, Optional, Text

# import regex
# import re
# from typing import Pattern

# import rasa.shared.utils.io
# import rasa.utils.io

# from rasa.engine.graph import ExecutionContext
# from rasa.engine.recipes.default_recipe import DefaultV1Recipe
# from rasa.engine.storage.resource import Resource
# from rasa.engine.storage.storage import ModelStorage
# from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
# from rasa.shared.constants import DOCS_URL_COMPONENTS
# from rasa.shared.nlu.training_data.message import Message


# @DefaultV1Recipe.register(
#     DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
# )
# class CustomTokenizer(Tokenizer):
#     """Creates features for entity extraction."""

#     @staticmethod
#     def supported_languages() -> Optional[List[Text]]:
#         """Determines which languages this component can work with.

#         Returns: A list of supported languages, or `None` to signify all are supported.
#         """
#         return ["id"]

#     @staticmethod
#     def get_default_config() -> Dict[Text, Any]:
#         """Returns the component's default config."""

#         # default config for base Tokenizer rasa
#         return {
#             # Flag to check whether to split intents
#             "intent_tokenization_flag": False,
#             # Symbol on which intent should be split
#             "intent_split_symbol": "_",
#             # Regular expression to detect tokens
#             "token_pattern": None,
#         }

#     def emoji_regex_pattern(self) -> Pattern:
#         """Returns regex to identify emojis."""
#         return re.compile(
#             "["
#             "\U0001F600-\U0001F64F"  # emoticons unicode range
#             "\U0001F300-\U0001F5FF"  # symbols & pictographs unicode range
#             "\U0001F680-\U0001F6FF"  # transport & map symbols unicode range
#             "\U0001F1E0-\U0001F1FF"  # flags (iOS) unicode range
#             "\U00002702-\U000027B0"
#             "\U000024C2-\U0001F251"
#             "\u200d"  # zero width joiner
#             "\u200c"  # zero width non-joiner
#             "]+",
#             flags=re.UNICODE,
#         )

#     def special_chars_pattern(self) -> Pattern:
#         """Returns regex to identify unneeded special chars"""
#         return re.compile("[!@#$%^&*()-+{}\\[\\]\\/<>~`'\"]", flags=re.ASCII)

#     def __init__(self, config: Dict[Text, Any]) -> None:
#         """Initialize the tokenizer."""
#         super().__init__(config)

#         self.emoji_pattern = self.emoji_regex_pattern()
#         self.special_chars_pattern_regex = self.special_chars_pattern()

#         if "case_sensitive" in self._config:
#             rasa.shared.utils.io.raise_warning(
#                 "The option 'case_sensitive' was moved from the tokenizers to the "
#                 "featurizers.",
#                 docs=DOCS_URL_COMPONENTS,
#             )

#     @classmethod
#     def create(
#         cls,
#         config: Dict[Text, Any],
#         model_storage: ModelStorage,
#         resource: Resource,
#         execution_context: ExecutionContext,
#     ) -> CustomTokenizer:
#         """Creates a new component (see parent class for full docstring)."""
#         # Path to the dictionaries on the local filesystem.
#         return cls(config)

#     def remove_emoji(self, text: Text) -> Text:
#         """Remove emoji if the full text, aka token, matches the emoji regex."""
#         match = self.emoji_pattern.fullmatch(text)

#         if match is not None:
#             # there is emoji in the text (token)
#             return ""

#         # there is no emoji in the text (token)
#         return text

#     def remove_special_chars(self, text: Text) -> Text:
#         return self.special_chars_pattern_regex.sub("", text)

#     def tokenize(self, message: Message, attribute: Text) -> List[Token]:
#         text = message.get(attribute)

#         # we need to use regex instead of re, because of
#         # https://stackoverflow.com/questions/12746458/python-unicode-regular-expression-matching-failing-with-some-unicode-characters

#         # remove 'not a word character' if
#         words = regex.sub(
#             # there is a space or an end of a string after it
#             r"[^\w#@&]+(?=\s|$)|"
#             # there is a space or beginning of a string before it
#             # not followed by a number
#             r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
#             # not in between numbers and not . or @ or & or - or #
#             # e.g. 10'000.00 or blabla@gmail.com
#             # and not url characters
#             r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
#             " ",
#             text,
#         ).split()

#         words = [self.remove_special_chars(
#             self.remove_emoji(w)) for w in words]

#         # filter non empty strings
#         words = [w for w in words if w]

#         # if we removed everything like smiles `:)`, use the whole text as 1 token
#         if not words:
#             words = [text]

#         tokens = self._convert_words_to_tokens(words, text)

#         return self._apply_token_pattern(tokens)

from __future__ import annotations
from typing import Any, Dict, List, Optional, Text

from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message
import re
from typing import Pattern


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class SpecialCharsRemovalTokenizer(Tokenizer):
    """Creates features for entity extraction."""

    @staticmethod
    def supported_languages() -> Optional[List[Text]]:
        """The languages that are not supported."""
        return ["en", "id"]

    def get_special_chars_pattern(self) -> Pattern:
        """Returns regex to identify unneeded special chars"""
        return re.compile("[!@#$%^&*()-+{}\\[\\]\\/<>~`'\" ]", flags=re.ASCII)

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            # This *must* be added due to the parent class.
            "intent_tokenization_flag": False,
            # This *must* be added due to the parent class.
            "intent_split_symbol": "_",
            # This is a, somewhat silly, config that we pass
            "only_alphanum": True,
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initialize the tokenizer."""
        super().__init__(config)
        self.special_chars_regex = self.get_special_chars_pattern()

    def remove_special_chars(self, text):
        return self.special_chars_regex.sub("", text)

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> SpecialCharsRemovalTokenizer:
        return cls(config)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = self.remove_special_chars(message.get(attribute))
        words = [w for w in text.split(" ") if w]

        # if we removed everything like smiles `:)`, use the whole text as 1 token
        if not words:
            words = [text]

        # the ._convert_words_to_tokens() method is from the parent class.
        tokens = self._convert_words_to_tokens(words, text)

        return self._apply_token_pattern(tokens)
