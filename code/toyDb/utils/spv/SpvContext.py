import dataclasses
from typing import Union, List, Dict
import os, json
import array
import io

"""
Courtesy: https://github.com/kristerw/spirv-tools
"""

def pack_bytes_to_bytes_stream(dataBytes):
  return io.BytesIO(dataBytes)

SpvGrammarPath = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  "./spirv.core.grammar.json"
)

@dataclasses.dataclass
class LiteralOperand:
  val: Union[str, int, float]

  def toTextRepr(self) -> str:
    return self.val

@dataclasses.dataclass
class RawLiteralOperand:
  """
  This occurs as a result of parsing LiteralContextDependentNumber,
  Since thorough parsing would require a TypeManager.
  """
  vals: List[int]

  def toTextRepr(self) -> str:
    return self.vals

@dataclasses.dataclass
class IdOperand:
  val: int

  def toTextRepr(self) -> str:
    return self.val

@dataclasses.dataclass
class EnumOperand:
  enumerant: int
  kind: str
  parameters: Union[
    List[Union[IdOperand, RawLiteralOperand, LiteralOperand, 'EnumOperand']],
    None
  ]

  def toTextRepr(self) -> str:
    result = f"{self.enumerant}"
    if self.parameters is not None:
      for operand in self.parameters:
        result += f"{operand.toTextRepr()} "
      
    return result

def getGrammarFromParentPtr(
    parent: Union['BasicBlock', 'Function', 'BinaryContext']
  ) -> Union[None, 'Grammar']:
  
  ref = parent
  while True:
    if ref is None:
      return None
    elif isinstance(ref, BasicBlock):
      ref = ref.parent
    elif isinstance(ref, Function):
      ref = ref.parent
    elif isinstance(ref, BinaryContext):
      ref = ref.grammar
      return ref
    else:
      assert(False)


@dataclasses.dataclass
class Instruction:
  opcode: int
  operands: List[Union[LiteralOperand, IdOperand]] = dataclasses.field(default_factory=list)
  parent: Union['BasicBlock', 'Function', 'BinaryContext', None] = None

  def resultId(self, grammar: 'Grammar') -> Union[int, None]:
    """Result id"""
    if grammar.hasResultId[self.opcode]:
      if grammar.hasTypeId[self.opcode]:
        return self.operands[1].val
      else:
        return self.operands[0].val
    else:
      return None

  def typeId(self, grammar: 'Grammar') -> Union[int, None]:
    """Result type Id"""
    if grammar.hasTypeId[self.opcode]:
      return self.operands[0].val
    else:
      return None
  
  def toTextRepr(self) -> str:
    grammar = getGrammarFromParentPtr(self.parent)

    if grammar is None:
      return f"{self.opcode} {self.operands}"
    
    opname = grammar.instFormatsByOpcode[self.opcode].opname
    result = ""
    offset = 0
    if self.resultId(grammar) is not None and self.typeId(grammar) is not None:
      result += f"%{self.resultId(grammar)} = {opname} %{self.typeId(grammar)} "
      offset = 2
    elif self.resultId(grammar) is not None and self.typeId(grammar) is None:
      result += f"%{self.resultId(grammar)} = {opname} "
      offset = 1
    else:
      result += f"{opname} "
    
    for operand in self.operands[offset:]:
      result += f"{operand.toTextRepr()} "
    
    return result
  

@dataclasses.dataclass
class BasicBlock:
  blockId: int
  insts: List[Instruction] = dataclasses.field(default_factory=list)
  parent: Union['Function', None] = None

  def toTextRepr(self) -> str:
    result = f"%{self.blockId} = OpLabel"
    for inst in self.insts:
      result += f"\n{inst.toTextRepr()}"
    
    return result

@dataclasses.dataclass
class Function:
  func: Instruction
  params: List[Instruction] = dataclasses.field(default_factory=list)
  basicBlocks: List['BasicBlock'] = dataclasses.field(default_factory=list)
  parent: Union['BinaryContext', None] = None

  def toTextRepr(self) -> str:
    result = f"{self.func.toTextRepr()}"
    for inst in self.params:
      result += f"\n{inst.toTextRepr()}"
    
    for bb in self.basicBlocks:
      result += f"\n{bb.toTextRepr()}"

    return result

@dataclasses.dataclass
class InstructionFormat:
  opname: str
  instrClass: str
  opcode: int
  operands: List[Dict[str, str]]

"""
Category:
BitEnum
ValueEnum
Id
Literal
Composite
"""

class Grammar:
  def __init__(self, spvGrammarPath=SpvGrammarPath):
    self.instFormatsByOpcode: Dict[int, 'InstructionFormat'] = {}
    self.opcode : Dict[str, int] = {}
    self.hasResultId : Dict[int, bool] = {}
    self.hasTypeId : Dict[int, bool] = {}

    # TODO: retrieve from grammar json
    self.grammarVersion = 0x00010600

    self.kindCategory: Dict[str, str] = {}
    self.kinds = {}

    self.instrClasses = set()

    with open(SpvGrammarPath, "r") as f:
      spvGrammarJson = json.load(f)

      for instDesc in spvGrammarJson["instructions"]:
        opname = instDesc["opname"]
        opcode = instDesc["opcode"]

        self.instFormatsByOpcode[opcode] = InstructionFormat(
          opname=opname,
          instrClass=instDesc["class"],
          opcode=opcode,
          operands=instDesc["operands"] if "operands" in instDesc else tuple()
        )
        if instDesc["class"] not in self.instrClasses:
          self.instrClasses.add(instDesc["class"])

        self.opcode[opname] = opcode

        self.hasResultId[opcode] = False
        self.hasTypeId[opcode] = False

        if "operands" in instDesc and len(instDesc["operands"]) > 0:
          if instDesc["operands"][0]["kind"] == "IdResultType":
            self.hasTypeId[opcode] = True
            assert(instDesc["operands"][1]["kind"] == "IdResult")
            self.hasResultId[opcode] = True
          elif instDesc["operands"][0]["kind"] == "IdResult":
            self.hasTypeId[opcode] = False
            self.hasResultId[opcode] = True

      for kindInst in spvGrammarJson["operand_kinds"]:
        self.kindCategory[kindInst["kind"]] = kindInst["category"]
        self.kinds[kindInst["kind"]] = kindInst
      
    
    self.blockTerminateInstOpCodes = (
      self.opcode["OpReturn"],
      self.opcode["OpReturnValue"],
      self.opcode["OpKill"],
      self.opcode["OpUnreachable"],
      self.opcode["OpTerminateInvocation"],
      self.opcode["OpBranch"],
      self.opcode["OpBranchConditional"],
      self.opcode["OpSwitch"]
    )


def getDefaultGrammar():
  return Grammar()

class ParseError(Exception):
  pass

@dataclasses.dataclass
class BinaryContext:
  # premable
  version: int
  generation: int
  idBound: int
  instSchema: int

  # global insts - TODO: separate them
  globalInsts : List[Instruction] = dataclasses.field(default_factory=list)

  # functions
  functions : List[Function] = dataclasses.field(default_factory=list)

  # grammar; used for dump things
  grammar: Union['Grammar', None] = None

  def toTextRepr(self) -> str:
    result = "; SPIR-V SpvContext Dump"
    result += f"\n; Version: {self.version:#x}"
    result += f"\n; Generation: {self.generation:#x}"
    result += f"\n; IdBound: {self.idBound}"
    result += f"\n; InstSchema: {self.instSchema}"

    for inst in self.globalInsts:
      result += f"\n{inst.toTextRepr()}"
    
    for func in self.functions:
      result += f"\n{func.toTextRepr()}"

    return result

class BinaryParser:
  def __init__(self):
    self.offset = 0
    self.totalLength = 0
    self.grammar: Union['Grammar', None] = None
    self.instLengthLeft: int = 0
    self.curInstOpcode: Union[int, None] = None
    self.parsedOperands: List = []
    self.instStreamBuffer: List = []

  def parse(self, grammar: 'Grammar', byteStream) -> 'BinaryContext':
    data = byteStream.read()
    if len(data) % 4 != 0:
      raise ParseError('File length is not divisible by 4')
    
    self.words = array.array('I', data)
    self.offset = 0
    self.grammar = grammar
    self.instLengthLeft = 0
    self.curInstOpcode = None
    self.parsedOperands: List = []

    ctx = self.parsePreamable(grammar)
    
    # init inst stream
    self.initInstStream()
    
    self.parseGlobalInsts(ctx)
    self.parseFunctions(grammar, ctx)

    return ctx

  def parseGlobalInsts(self, ctx: 'BinaryContext'):
    # global instructions
    while not self.instStreamEnds():
      if self.peekInstStream().opcode == self.grammar.opcode["OpFunction"]:
        break

      ctx.globalInsts.append(self.consumeInstStream(ctx))
  
  def parseFunctions(self, grammar, ctx: 'BinaryContext'):
    while not self.instStreamEnds():
      assert(self.peekInstStream().opcode == self.grammar.opcode["OpFunction"])
      funcInst = self.consumeInstStream()

      newFunc = Function(funcInst, parent=ctx)
      newFunc.func.parent = newFunc

      # parse params
      while not self.instStreamEnds():
        if self.peekInstStream().opcode != self.grammar.opcode["OpFunctionParameter"]:
          break

        newFunc.params.append(self.consumeInstStream(newFunc))
      
      # parse basic blocks
      self.parseBasicBlocks(grammar, newFunc)

      endInst = self.consumeInstStream()
      if endInst.opcode != self.grammar.opcode["OpFunctionEnd"]:
        raise ParseError(f"Ill-formed function while parsing")
      
      ctx.functions.append(newFunc)

  
  def parseBasicBlocks(self, grammar, newFunc: 'Function'):
    curBasicBlock = None

    while True:
      nextOpcode = self.peekInstStream().opcode

      if nextOpcode == self.grammar.opcode["OpFunctionEnd"]:
        if curBasicBlock is not None:
          raise ParseError(f"Ill-formed basic block while parsing")
        
        break
      elif nextOpcode == self.grammar.opcode["OpLabel"]:
        if curBasicBlock is not None:
          raise ParseError(f"Ill-formed basic block while parsing")
        
        labelInst = self.consumeInstStream()
        bbIdx = labelInst.resultId(grammar)
        assert(bbIdx is not None)

        curBasicBlock = BasicBlock(bbIdx, [], newFunc)
      elif nextOpcode in self.grammar.blockTerminateInstOpCodes:
        nextInst = self.consumeInstStream(curBasicBlock)
        curBasicBlock.insts.append(nextInst)
        
        newFunc.basicBlocks.append(curBasicBlock)
        curBasicBlock = None
      else:
        if curBasicBlock is None:
          raise ParseError(f"Ill-formed basic block while parsing")

        nextInst = self.consumeInstStream(curBasicBlock)
        curBasicBlock.insts.append(nextInst)
  
  def initInstStream(self):
    self.instStreamBuffer = []

  def instStreamEnds(self):
    return len(self.instStreamBuffer) == 0 and self.eos()

  def peekInstStream(self) -> Instruction:
    if len(self.instStreamBuffer) > 0:
      return self.instStreamBuffer[0]
    else:
      self.instStreamBuffer = [self.parseInstruction()]
      return self.instStreamBuffer[0]
  
  def consumeInstStream(
      self,
      newParent: Union['BinaryContext', 'Function', 'BasicBlock', None]=None
    ) -> Instruction:
    if len(self.instStreamBuffer) > 0:
      newInst = self.instStreamBuffer.pop(0)
    else:
      newInst = self.parseInstruction()
    
    if newParent is not None:
      newInst.parent = newParent
    
    return newInst
  
  def parsePreamable(self, grammar) -> 'BinaryContext':
    if len(self.words) < 5:
      raise ParseError("File length shorter than header size")

    magic = self.words[0]
    if magic != 0x07230203:
      self.words.byteswap()
      magic = self.words[0]
      if magic != 0x07230203:
        raise ParseError("Incorrect magic: " + format(magic, '#x'))

    version = self.words[1]
    if version >= self.grammar.grammarVersion:
        raise ParseError(f"SPIR-V version {version} higher than grammar version")

    ctx = BinaryContext(
      version=version,
      generation=self.words[2],
      idBound=self.words[3],
      instSchema=self.words[4],
      grammar=grammar
    )

    self.offset = 5

    return ctx

  def eoi(self):
    return self.instLengthLeft == 0

  def eos(self):
    return self.offset == len(self.words)

  def consumeNextInstWord(self) -> int:
    if not self.eos() or self.eoi():
      word = self.words[self.offset]
      self.offset += 1
      self.instLengthLeft -= 1
      return word
    else:
      raise ParseError("No further word to consume")

  def parseIdOperand(self) -> IdOperand:
    word = self.consumeNextInstWord()
    newId = IdOperand(word)
    return newId
  
  def parseEnumOperand(self, enumKind: str) -> EnumOperand:
    word = self.consumeNextInstWord()
    newEnum = EnumOperand(enumerant=word, kind=enumKind, parameters=None)

    # find proper enumerant
    paramKinds = []
    if self.grammar.kindCategory[enumKind] == "ValueEnum":
      enumerantDesc = None
      for idx, descCand in enumerate(self.grammar.kinds[enumKind]["enumerants"]):
        if descCand["value"] == word:
          enumerantDesc = descCand
    
      if "parameters" in enumerantDesc:
        paramKinds += enumerantDesc["parameters"]
      if enumerantDesc is None:
        raise ParseError(f"Unknown enumerant value {word} for {enumKind}")

    elif self.grammar.kindCategory[enumKind] == "BitEnum":
      # If there are multiple following operands indicated, they are ordered:
      # Those indicated by smaller-numbered bits appear first.

      # test for selected
      selected = []
      for idx, enumerantDesc in enumerate(self.grammar.kinds[enumKind]["enumerants"]):
        if word & int(enumerantDesc["value"], base=16) != 0:
          selected.append(idx)

      for idx in selected:
        if "parameters" in self.grammar.kinds[enumKind]["enumerants"][idx]:
          paramKinds += self.grammar.kinds[enumKind]["enumerants"][idx]["parameters"]

    if len(paramKinds) > 0:
      parameters = []
      for idx, operandDesc in enumerate(paramKinds):
        if "quantifier" not in operandDesc:
          # must consume
          parameters += self.parseOperand(operandDesc["kind"])
        elif operandDesc["quantifier"] == "?":
          # could either consume or give up
          if self.eoi():
            break
          
          parameters += self.parseOperand(operandDesc["kind"])
        elif operandDesc["quantifier"] == '*':
          # could consume 0 or many; should be the last one
          # or we must have dedicated parsing logic
          assert(idx == len(paramKinds) - 1)

          while not self.eoi():
            parameters += self.parseOperand(operandDesc["kind"])

      newEnum.parameters = parameters

    return newEnum
  
  def parseLiteralIntegerOperand(self) -> LiteralOperand:
    word = self.consumeNextInstWord()
    newLiteral = LiteralOperand(word)
    return newLiteral

  def parseOperand(self, kind) -> List[Union[IdOperand, LiteralOperand, EnumOperand, RawLiteralOperand]]:
    """We need peek functionality here"""
    if self.grammar.kindCategory[kind] == "Id":
      return [self.parseIdOperand()]
    elif self.grammar.kindCategory[kind] == "BitEnum":
      return [self.parseEnumOperand(kind)]
    elif self.grammar.kindCategory[kind] == "ValueEnum":
      return [self.parseEnumOperand(kind)]
    elif kind in (
        "LiteralInteger", "LiteralSpecConstantOpInteger", "LiteralExtInstInteger"
      ):
      return [self.parseLiteralIntegerOperand()]
    elif kind == "LiteralContextDependentNumber":
      # this is only in OpConstant and OpSpecConstant,
      # so a cheap way to parse is to eat all the rest of the word
      vals = []
      while not self.eoi():
        vals.append(self.consumeNextInstWord())
      return [RawLiteralOperand(vals)]
    elif kind == "LiteralString":
      return [self.parseLiteralString()]
    elif kind == "PairLiteralIntegerIdRef":
      first = self.parseLiteralIntegerOperand()
      second = self.parseIdOperand()
      return [first, second]
    elif kind == "PairIdRefLiteralInteger":
      first = self.parseIdOperand()
      second = self.parseLiteralIntegerOperand()
      return [first, second]
    elif kind == "PairIdRefIdRef":
      first = self.parseIdOperand()
      second = self.parseIdOperand()
      return [first, second]
    else:
      raise ParseError(f"Unhandled operand kind {kind}")

  def parseLiteralString(self):
    """Parse one LiteralString."""
    result = []
    while True:
      word = self.consumeNextInstWord()
      for _ in range(4):
        octet = word & 255
        if octet == 0:
          return LiteralOperand(val=''.join(result))
        result.append(chr(octet))
        word >>= 8

  def parseInstruction(self) -> Instruction:
    word = self.consumeNextInstWord()
    opcode = word & 0xFFFF
    instLength = (word >> 16)
    
    self.curInstOpcode = opcode
    self.instLengthLeft = instLength - 1
    assert(self.instLengthLeft >= 0)

    if opcode not in self.grammar.instFormatsByOpcode:
      raise ParseError(f"Unknown opcode {opcode}")
    
    instFormat = self.grammar.instFormatsByOpcode[opcode]
    self.parsedOperands = []

    for idx, operandDesc in enumerate(instFormat.operands):
      if "quantifier" not in operandDesc:
        # must consume
        self.parsedOperands += self.parseOperand(operandDesc["kind"])
      elif operandDesc["quantifier"] == "?":
        # could either consume or give up
        if self.eoi():
          break
        
        self.parsedOperands += self.parseOperand(operandDesc["kind"])
      elif operandDesc["quantifier"] == '*':
        # could consume 0 or many; should be the last one
        # or we must have dedicated parsing logic
        assert(idx == len(instFormat.operands) - 1)

        while not self.eoi():
          self.parsedOperands += self.parseOperand(operandDesc["kind"])
    
    if not self.eoi():
      raise ParseError("Instruction too short")

    return Instruction(opcode, self.parsedOperands)

if __name__ == '__main__':
  pass