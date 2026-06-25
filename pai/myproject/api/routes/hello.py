"""Route handlers for greeting endpoints."""

from fastapi import APIRouter

from pai.myproject.metrics.compute import char_count, word_count
from pai.myproject.schemas.hello import HelloRequest, HelloResponse
from pai.myproject.services.hello import say_hello

router = APIRouter()


@router.post("/v1/hello", response_model=HelloResponse)
def hello(request: HelloRequest) -> HelloResponse:
    """Return a greeting and basic text stats for the supplied name.

    Parameters
    ----------
    request : HelloRequest
        Validated request payload containing ``name`` and optional ``shout``.

    Returns
    -------
    HelloResponse
        Greeting message with word and character counts.
    """
    greeting = say_hello(request.name, shout=request.shout)
    return HelloResponse(
        greeting=greeting,
        word_count=word_count(greeting),
        char_count=char_count(greeting),
    )
