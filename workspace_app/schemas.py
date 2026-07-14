from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=500)


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class FileWrite(BaseModel):
    content: str


class RunCreate(BaseModel):
    workspaceId: str
    task: str = Field(min_length=1, max_length=20_000)


class FeedbackUpsert(BaseModel):
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=5000)
