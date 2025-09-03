class TrieNode:
    def __init__(self):
        # map character -> TrieNode
        self.children = {}
        # True if some word ends exactly here
        self.is_end = False
        # how many words pass through this node (including those ending here)
        self.pass_count = 0
        # how many words end exactly here (useful but optional)
        self.word_count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        node.pass_count += 1
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.pass_count += 1
        node.is_end = True
        node.word_count += 1

    def search(self, word: str) -> bool:
        """Return True if the full word exists in the trie."""
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        """Return True if any word starts with this prefix."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True
    
    def find_stem_suffix(self, word: str):
        """Return (stem, suffix) based on branching point."""
        node = self.root
        stem = ""
        last_branch_index = 0

        for i, ch in enumerate(word):
            node = node.children.get(ch)
            if node is None:
                break

            stem += ch
            # branching point if multiple children OR multiple words end here
            if len(node.children) > 1 or node.word_count > 1:
                last_branch_index = i + 1  # record split after this char

        return word[:last_branch_index], word[last_branch_index:]
    
class SuffixTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        node.pass_count += 1
        # insert the word reversed
        for ch in reversed(word):
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.pass_count += 1
        node.is_end = True
        node.word_count += 1

    def find_stem_suffix(self, word: str):
        """Return (stem, suffix) using suffix trie branching."""
        node = self.root
        reversed_word = list(reversed(word))
        last_branch_index = 0
        suffix = ""

        for i, ch in enumerate(reversed_word):
            node = node.children.get(ch)
            if node is None:
                break

            suffix = ch + suffix  # build suffix in correct order
            if len(node.children) > 1 or node.word_count > 1:
                last_branch_index = i + 1  # split here

        return word[:-last_branch_index], word[-last_branch_index:]


if __name__ == "__main__":
    words = ["goes", "kites", "lives"]
    ST = SuffixTrie()
    for w in words:
        ST.insert(w)

    for w in words:
        stem, suffix = ST.find_stem_suffix(w)
        print(f"{w} = {stem} + {suffix}")



