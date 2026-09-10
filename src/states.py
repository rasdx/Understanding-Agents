# Reference examples of how to define LangGraph-style shared state in Python.
# TypedDict: simple typed dictionary with no runtime validation.
# Pydantic BaseModel: same fields with descriptions and checks when data is created.
# Dataclass: a lightweight class holding name, abbreviation, capital, population, and area.

import os
from typing import TypedDict


# 1) TypeDict approach for defining a state
class State(TypedDict):
    name: str
    abbreviation: str
    capital: str
    population: int
    area: float 

# 2) Pydantic approach for defining a state data validation and type checking at runtime
from pydantic import BaseModel, Field

class StateModel(BaseModel):
    name: str = Field(..., description="The name of the state")
    abbreviation: str = Field(..., description="The abbreviation of the state")
    capital: str = Field(..., description="The capital of the state")
    population: int = Field(..., description="The population of the state")
    area: float = Field(..., description="The area of the state")

# 3) Dataclass approach for defining a state
from dataclasses import dataclass
@dataclass
class StateDataClass:
    name: str
    abbreviation: str
    capital: str
    population: int
    area: float
