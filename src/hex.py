from __future__ import annotations  # Necessary to use the class as a type annotation in its own members.

import copy
from typing import Optional  # For T | None annotations.
from typing import Union

import math

from pixel import PixelCoord

move_vectors = {
    **dict.fromkeys(["w_bishop", "b_bishop", "r_bishop"], [
        (1, 1, -2),(-1, 2, -1),
        (-2, 1, 1),(-1, -1, 2),
        (1, -2, 1),(2, -1, -1)
    ]),

    **dict.fromkeys(["w_rook", "b_rook","r_rook"], [
        (0, 1, -1), (0, -1, 1),
        (1, 0, -1), (-1, 0, 1),
        (1, -1, 0), (-1, 1, 0)
    ]),

    **dict.fromkeys(["w_king", "b_king","r_king"], [
        (0, 1, -1), (0, -1, 1),
        (1, 0, -1), (-1, 0, 1),
        (1, -1, 0), (-1, 1, 0),
        (2, -1, -1), (-2, 1, 1),
        (-1, 2, -1), (1, -2, 1),
        (-1, -1, 2), (1, 1, -2)
    ]),

    **dict.fromkeys(["w_queen", "b_queen",  "r_queen"], [
        (0, 1, -1), (0, -1, 1),
        (1, 0, -1), (-1, 0, 1),
        (1, -1, 0), (-1, 1, 0),
        (2, -1, -1), (-2, 1, 1),
        (-1, 2, -1), (1, -2, 1),
        (-1, -1, 2), (1, 1, -2)
    ]),

    **dict.fromkeys(["w_knight", "b_knight", "r_knight"], [
        (2, 1, -3), (3, -1, -2),
        (-1, 3, -2), (1, 2, -3),
        (-2, 3, -1), (-3, 2, 1),
        (-3, 1, 2), (-2, -1, 3),
        (-1, -2, 3), (1, -3, 2),
        (2, -3, 1), (3, -2, -1)
    ]),

    "w_pawn": [
        (0, 1, -1),
        (-1, 2, -1),
        (1, 1, -2)

    ],

    "b_pawn": [
        (1, -1, 0),
        (2, -1, -1),
        (1, -2, 1)
    ],

    "r_pawn": [
        (-1, 0, 1),
        (-1, -1, 2),
        (-2, 1, 1)
    ]
}


class HexCoord:
    """
    A wrapper class over the concept of a hexagonal coordinate.
    The class provides operator overloads such as +, -, * and /, as well as == and !=.
    The class also provides native function overloads like round(), hash() and abs().
    This allows a Pythonic interface with hexagonal geometry.
    """

    def __init__(self, p: float, q: float, r: float):
        self.p, self.q, self.r = p, q, r

    def __add__(self, other: HexCoord) -> HexCoord:
        """Vector addition between two HexCoords."""
        return HexCoord(self.p + other.p, self.q + other.q, self.r + other.r)

    def __sub__(self, other: HexCoord) -> HexCoord:
        """Vector subtraction between two HexCoords."""
        return HexCoord(self.p - other.p, self.q - other.q, self.r - other.r)

    def __mul__(self, other: float) -> HexCoord:
        """Vector multiplication by a scalar"""
        return HexCoord(self.p * other, self.q * other, self.r * other)

    def __truediv__(self, other: float) -> HexCoord:
        """Vector division by a scalar."""
        return HexCoord(self.p / other, self.q / other, self.r / other)

    def __eq__(self, other: HexCoord) -> bool:
        """Component-wise equality check."""
        if type(other) is not type(self):
            return False

        return self.p == other.p and self.q == other.q and self.r == other.r

    def __round__(self, n=None) -> HexCoord:
        """Round to the HexCoord that this coordinate is in."""
        rp = round(self.p)
        rq = round(self.q)
        rr = round(self.r)

        p_diff = abs(self.p - rp)
        q_diff = abs(self.q - rq)
        r_diff = abs(self.r - rr)

        diff_list = [p_diff, q_diff, r_diff]

        if max(diff_list) == p_diff:
            rp = -(rq + rr)
        elif max(diff_list) == q_diff:
            rq = -(rp + rr)
        else:
            rr = -(rp + rq)

        return HexCoord(rp, rq, rr)

    def __str__(self) -> str:
        """A user-friendly representation."""
        return f"({self.p}, {self.q}, {self.r})"

    def __repr__(self):
        """A coder's representation."""
        return f"HexCoord({self.p}, {self.q}, {self.r})"

    def __hash__(self) -> int:
        """Unique hashing that takes into account order of values."""
        return hash((self.p, self.q, self.r))

    def __abs__(self) -> HexCoord:
        """
        Map each coordinate to their absolute values.
        Not guaranteed to be a valid Hex Coordinate geometrically afterwards.
        """
        return HexCoord(abs(self.p), abs(self.q), abs(self.r))

    def __iter__(self) -> iter:
        
        """Return an iterator over the coordinate's components."""
        return iter([self.p, self.q, self.r])

    def mag(self) -> float:
        return math.sqrt(self.p**2 + self.q**2 + self.r**2)


class HexCell:
    """A class to group a coordinate on the board and it's corresponding state."""
    def __init__(self, coord: HexCoord, state=None):
        self.coord = coord
        self.state = state


class HexMap:
    """
    A wrapper class over the game board.
    It provides methods to make moves, generate common boards, moves from a coordinate and detect check / checkmate.
    This class overloads operators like `in` and `[]`.
    It also overloads native functions like iter() to allow iteration over a board.
    This provides an easier interface to the board.
    """

    def __init__(self, cells=None):
        if cells is None:
            cells = dict()
        self.cells: dict[int, HexCell] = cells
        self.coord_to_cell_registry: dict[HexCoord, int] = dict()
        self.ply: int = 0
        self.__hash__ = self.__str__

    def __iter__(self):
        """Return an iterator over the `HexCoord`s in the map."""
        return iter(self.cells.values())

    def __getitem__(self, item: Union[int, HexCoord]) -> Optional[str]:
        """Get the state at a specific `HexCoord` in the map."""
        if type(item) is HexCoord:
            return self.cells[self.coord_to_cell_registry[item]].state
        elif type(item) is int:
            return self.cells[item].state

    def __setitem__(self, key: Union[int, HexCoord], value: Optional[str]):
        """Set the state at a specific `HexCoord` in the map."""
        if type(key) is HexCoord:
            self.cells[self.coord_to_cell_registry[key]].state = value
        elif type(key) is int:
            self.cells[key].state = value

    def __contains__(self, item: HexCoord) -> bool:
        """
        Check if a `HexCoord` is within the map.
        This is useful for out of board checks.
        """
        if type(item) is HexCoord:
            return item in self.coord_to_cell_registry.keys()
        elif type(item) is int:
            return item in self.cells.keys()

    def __str__(self) -> str:
        FEN_dict = {
            None: "x",

            "w_pawn": "Pw",
            "b_pawn": "Pb",
            "r_pawn": "Pr",

            "w_rook": "Rw",
            "b_rook": "Rb",
            "r_rook": "Rr",

            "w_king": "Kw",
            "b_king": "Kb",
            "r_king": "Kr",

            "w_bishop": "Bw",
            "b_bishop": "Bb",
            "r_bishop": "Br",

            "w_queen": "Qw",
            "b_queen": "Qb",
            "r_queen": "Qr",

            "w_knight": "Nw",
            "b_knight": "Nb",
            "r_knight": "Nr"
        }

        hash_str = ""
        for i in range(91):
            hash_str += FEN_dict[self[i]]
        return hash_str

    @staticmethod
    def from_radius(radius: int) -> HexMap:
        """
        Generate a `HexMap` of a certain radius.
        This provides a useful lemma to build the Glinski variant.

        """
        hex_map = HexMap()
        i = 0
        for p in range(-radius, radius + 1):
            for q in range(-radius, radius + 1):
                for r in range(-radius, radius + 1):
                    if p + q + r == 0:
                        coord = HexCoord(p, q, r)
                        hex_map.coord_to_cell_registry[copy.deepcopy(coord)] = i
                        hex_map.cells[i] = HexCell(coord)
                        i += 1
        return hex_map

    @staticmethod
    def from_glinski() -> HexMap:
        """Generate a `HexMap` of Glinski's Hexagonal Variant."""

        # The coordinates of every piece type in the Glinski variant.
        glinski_pos: dict[str, list[tuple]] = {

            "w_pawn": [(-n, -2, n + 2) for n in range(4)] + [(n, -n - 2, 2) for n in range(4)],
            "w_knight": [(-1, -3, 4), (1, -4, 3)],
            "w_rook": [(-2, -3, 5), (2, -5, 3)],
            "w_bishop": [(0, -3, 3), (0, -4, 4), (0, -5, 5)],
            "w_queen": [(-1, -4, 5)],
            "w_king": [(1, -5, 4)],

            "b_pawn": [(-2-n, 2, +n ) for n in range(4)] + [(-2, 2 +n , -n) for n in range(4)],
            "b_bishop": [(-5, 5, 0), (-4, 4, 0), (-3, 3, 0)],
            "b_queen": [(-4, 5, -1)],
            "b_king": [( -5, 4, 1)],
            "b_rook": [( -3, 5, -2), ( -5, 3, 2)],
            "b_knight": [(-4, 3, 1), (-3, 4, -1)],
            
            "r_pawn": [( 2, +n,-2-n ) for n in range(4)] + [( 2 +n, -n, -2) for n in range(4)],
            "r_bishop": [(5, 0, -5), ( 4, 0, -4), ( 3, 0, -3)],
            "r_king": [( 5, -1, -4)],
            "r_queen": [(  4, 1, -5)],
            "r_rook": [(  5, -2, -3), (  3, 2, -5)],
            "r_knight": [( 3, 1, -4), ( 4, -1, -3)],
            
        }

        # Generate an initial foundation board.
        initial_map: HexMap = HexMap.from_radius(5)

        # Add all the pieces onto the board, following Glinski layout.
        for key, pos_list in glinski_pos.items():
            for pos in pos_list:
                initial_map[HexCoord(*pos)] = key

        return initial_map

    def generate_moves(self, start: HexCoord) -> list[HexCoord]:

        """
        Generates all moves from a specified start coord.
        It travels along predefined vectors and checks certain conditions about whether it should stop.
        """

        start_state: Optional[str] = self[start]

        # If there is nothing at the coord, there are no moves logically available to make.
        if start_state is None:
            return []

        # A piece can always move back to where it started.
        valid_moves: list[HexCoord] = [start]

        for ray in move_vectors[start_state]:

            # Convert the 3-tuple into a `HexCoord` to take advantage of its overloaded operators and methods.
            ray: HexCoord = HexCoord(*ray)
            curr_hex: HexCoord = start

            while True:
                # Travel one along the vector.
                curr_hex += ray

                # Is the coord I'm at out of bounds? If so, stop moving along this line.
                if curr_hex not in self:
                    break

                # If the piece at the coord is the same colour as me, stop moving along this line.
                if self[curr_hex] is not None and self[curr_hex][0] == start_state[0]:
                    break

                # If the king is in check after this move:
                if self.is_king_checked_after_move(start_state[0], start, curr_hex):
                    # King, Pawn and Knight are not sliding pieces, so they must check the next vector immediately.
                    if start_state[2:] in ["king", "pawn", "knight"]:
                        break
                    # Every other piece is a sliding piece, so we can still check along this line for moves.
                    else:
                        continue

                # Special handling for the quirks of the pawn pieces
                if start_state.endswith("pawn"):
                    offset: HexCoord = curr_hex - start
                    if start_state[0] == "w":

                        # If the offset is a 'diagonal' attack move but there's nothing there to attack:
                        if offset in [HexCoord(-1, 2, -1), HexCoord(1, 1, -2)] and self[curr_hex] is None:
                            break

                        # If the offset is a 'forward' normal move but there's an enemy piece in the way:
                        if offset == HexCoord(0, 1, -1) and self[curr_hex] is not None:
                            break
                    elif start_state[0] == "b":
                        
                        # If the offset is a 'diagonal' attack move but there's nothing there to attack:
                        if offset in [HexCoord(2, -1, -1), HexCoord(1, -2, 1)] and self[curr_hex] is None:
                            break

                        # If the offset is a 'forward' normal move but there's an enemy piece in the way:
                        if offset == HexCoord(1, -1, 0) and self[curr_hex] is not None:
                            break
                    elif start_state[0] == "r":

                        # If the offset is a 'diagonal' attack move but there's nothing there to attack:
                        if offset in [HexCoord(-2, 1, 1), HexCoord(-1, -1, 2)] and self[curr_hex] is None:
                            break

                        # If the offset is a 'forward' normal move but there's an enemy piece in the way:
                        if offset == HexCoord(-1, 0, 1) and self[curr_hex] is not None:
                            break

                # The move passed all checks, so add it to the valid moves list.
                valid_moves.append(curr_hex)

                # If this is true, it must be the case that there is an enemy piece, so we can't travel any further
                # along this vector.
                if self[curr_hex] is not None:
                    break

                # King, Pawn and Knight can only move along their vectors once: they are not sliding pieces.
                # This will run on the first loop of any vector, stopping sliding behaviour for them.
                if start_state[2:] in ["king", "pawn", "knight"]:
                    break

        return valid_moves

    def cells_with_state_col(self, color: str) -> list[HexCell]:
        """Return all cells that have a piece of specified colour."""
        valid_cells: list[HexCell] = []
        for cell in self:
            if cell.state is not None and cell.state.startswith(color):
                valid_cells.append(cell)
        return valid_cells

    def moves_for_col(self, color: str) -> tuple[HexCoord, HexCoord]:

        for cell in self.cells_with_state_col(color):
            for coord in self.generate_moves(cell.coord):
                if cell.coord != coord:
                    yield cell.coord, coord

    def make_move(self, start: HexCoord, end: HexCoord):

        """Performs the move from `start` to `end`. Handles ply incrementing and piece movement."""
        if start == end:
            return
        
        self[end] = self[start]
        self[start] = None
        self.ply += 1

    def is_king_checked(self, color: str) -> bool:
        """Checks if a king of specified colour is in check right now."""

        # The coordinate that the king is on must be found, to check enemy moves against.
        king_coords: Optional[HexCoord] = None
        for cell in self:
            if cell.state == f"{color}_king":
                king_coords = cell.coord

        # Iterate over the cells dictionary, over key-pair values.
        for coord, cell in self.cells.items():

            # If there is nothing at that cell, there is no need to check if it can threaten the king.
            if cell.state is None:
                continue

            # If the piece is of the same colour as the king, it is certain not to threaten it.
            if cell.state[0] == color:
                continue

            # This piece must certainly now be an enemy piece, so iterate over its 'move vectors':
            for ray in move_vectors[cell.state]:
                # Convert the 3-tuple into a `HexCoord` to take advantage of its overloaded operators and methods.
                ray: HexCoord = HexCoord(*ray)
                curr_hex: HexCoord = cell.coord

                while True:

                    # Travel one along the vector.
                    curr_hex += ray

                    # Is the coord I'm at out of bounds? If so, stop moving along this line.
                    if curr_hex not in self:
                        break

                    # If the piece at the coord is the same colour as me, stop moving along this line.
                    if self[curr_hex] is not None and self[curr_hex][0] == cell.state[0]:
                        break

                    # Special handling for the quirks of the pawn pieces
                    if cell.state.endswith("pawn"):
                        offset: HexCoord = curr_hex - cell.coord

                        # We only need to check that the offset is an attack, since pawns cannot threaten forwards.
                        if cell.state[0] == "w":
                            if offset not in [HexCoord(-1, 2, -1), HexCoord(1, 1, -2)]:
                                break
                        elif cell.state[0] == "b":
                            if offset not in (HexCoord(2, -1, -1), HexCoord(1, -2, 1)):
                                break
                        elif cell.state[0] == "r":
                            if offset not in (HexCoord(-2, 1, 1), HexCoord(-1, -1, 2)):
                                break

                    # The move passed all checks, so if it threatens the king, the king is in check.
                    if curr_hex == king_coords:
                        return True

                    # If this is true, it must be the case that its an enemy piece other than the king, so we can't
                    # travel any further along this vector.
                    elif self[curr_hex] is not None:
                        break

                    # King, Pawn and Knight can only move along their vectors once: they are not sliding pieces.
                    # This will run on the first loop of any vector, stopping sliding behaviour for them.
                    elif cell.state[2:] in ["king", "pawn", "knight"]:
                        break

        # The king is not in check.
        return False

    def is_king_checkmated(self, color: str) -> bool:
        """Checks if a king of specified colour is checkmated."""

        # If the king isn't even checked, there's no need checking for checkmate.
        if self.is_king_checked(color):

            # If none of the pieces can make moves other than move back to start, that must be because they don't get
            # the king out of check. Hence, the king is helpless and checkmated.
            if all(len(self.generate_moves(cell.coord)) == 1 for cell in self.cells_with_state_col(color)):
                return True
        return False

    def is_king_checked_after_move(self, color: str, start: HexCoord, end: HexCoord) -> bool:
        """Checks if a king of specified colour will be in check after a move."""

        prev_state: Optional[str] = self[end]

        self.make_move(start, end)
        result: bool = self.is_king_checked(color)
        self.make_move(end, start)

        self[end] = prev_state
        return result


class HexPixelAdapter:
    """A class which provides helper methods to convert between `PixelCoord`s and `HexCoord`s easily."""
    def __init__(self, dimensions: PixelCoord, origin: PixelCoord, hex_radius: float):
        self.dimensions: PixelCoord = dimensions
        self.origin: PixelCoord = origin
        self.hex_radius: float = hex_radius

    def hex_to_pixel(self, coord: HexCoord) -> PixelCoord:
        """Converts from a `HexCoord` to a `PixelCoord`."""
        x: float = self.hex_radius * 1.5 * coord.p + self.origin.x
        y: float = self.hex_radius * (math.sqrt(3) * 0.5 * coord.p + math.sqrt(3) * coord.r) + self.origin.y

        return PixelCoord(x, y)

    def pixel_to_hex(self, coord: PixelCoord) -> HexCoord:
        """Converts from a `PixelCoord` to a `HexCoord`."""
        
        coord -= self.origin

        p: float = 2 / 3 * coord.x / self.hex_radius
        r: float = (-1 / 3 * coord.x + math.sqrt(3) / 3 * coord.y) / self.hex_radius

        return HexCoord(p, -p - r, r)

    def get_vertices(self, coord: HexCoord) -> list[PixelCoord]:
        """Gets the `PixelCoord` vertices of a hex at any `HexCoord`."""
        x, y = self.hex_to_pixel(coord)
        angle: float = math.pi / 3

        return [PixelCoord(
            self.hex_radius * math.cos(angle * i) + x,
            self.hex_radius * math.sin(angle * i) + y
        ) for i in range(6)]
