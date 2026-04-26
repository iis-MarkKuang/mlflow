from typing import TypeVar, Iterator, Callable, Optional

T = TypeVar("T")


class PagedList(list[T]):
    """
    Wrapper class around the base Python `List` type. Contains an additional `token`  string
    attribute that can be passed to the pagination API that returned this list to fetch additional
    elements, if any are available
    """

    def __init__(self, items: list[T], token):
        super().__init__(items)
        self.token = token

    def to_list(self):
        return list(self)


class PagedIterator(Iterator[T]):
    """
    Iterator that lazily fetches pages of results from a pagination API.
    """

    def __init__(
        self,
        first_page_func: Callable[[Optional[str]], PagedList[T]],
        max_items: Optional[int] = None,
    ):
        """
        Args:
            first_page_func: A function that takes a page token and returns a PagedList.
                To fetch the first page, pass None as the page token.
            max_items: Maximum number of items to yield in total, or None for no limit.
        """
        self._first_page_func = first_page_func
        self._max_items = max_items
        self._current_page: Optional[PagedList[T]] = None
        self._items_yielded = 0
        self._page_index = 0
        self._item_index = 0

    def __iter__(self) -> "PagedIterator[T]":
        return self

    def __next__(self) -> T:
        # Check if we've already yielded the maximum number of items
        if self._max_items is not None and self._items_yielded >= self._max_items:
            raise StopIteration

        # If we don't have a current page or are out of items in the current page, fetch the next page
        while self._current_page is None or self._item_index >= len(self._current_page):
            if self._page_index == 0:
                # Fetch the first page
                self._current_page = self._first_page_func(None)
            else:
                # Check if there's a next page to fetch
                if self._current_page.token is None:
                    raise StopIteration
                # Fetch the next page
                self._current_page = self._first_page_func(self._current_page.token)
            
            self._page_index += 1
            self._item_index = 0
            
            # If we got an empty page, we should check if we need to fetch more
            if len(self._current_page) == 0:
                if self._current_page.token is None:
                    raise StopIteration
                # Continue loop to fetch next page

        # Get the next item
        item = self._current_page[self._item_index]
        self._item_index += 1
        self._items_yielded += 1
        return item
