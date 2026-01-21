module Main exposing (..)

import Browser
import Browser.Navigation as Nav
import Dict exposing (Dict)
import File exposing (File)
import Html exposing (..)
import Html.Attributes exposing (..)
import Html.Events exposing (..)
import Http
import Json.Decode as D
import Json.Encode as E
import Set exposing (Set)
import Url exposing (Url)
import Url.Parser as Parser exposing ((</>), (<?>))
import Url.Parser.Query as Query



-- MAIN


main : Program () Model Msg
main =
    Browser.application
        { init = init
        , update = update
        , subscriptions = \_ -> Sub.none
        , view = view
        , onUrlChange = UrlChanged
        , onUrlRequest = LinkClicked
        }



-- MODEL


type alias Model =
    { key : Nav.Key
    , url : Url
    , page : Page
    , error : Maybe String

    -- Project setup form
    , csvFile : Maybe File
    , filterQuery : String
    , groupBy : String
    , imgPath : String
    , primaryKey : String
    , rootDpath : String
    , sam2Model : String
    , device : String
    , pointsPerCrop : String
    , cropsNLayers : String
    , predIouThresh : String
    , stabilityScoreThresh : String
    , cropsNmsThresh : String

    -- Project state
    , projectId : Maybe String
    , columns : List String
    , groupCount : Int
    , rowCount : Int

    -- Groups
    , groups : List GroupSummary
    , selectedGroupKey : Maybe String

    -- Sampling
    , nRefFrames : String
    , seed : String

    -- Frames
    , frames : List FrameSummary
    , currentFrameIndex : Int

    -- Masks
    , masks : List MaskMeta
    , selectedMasks : Dict Int Int -- mask_id -> label
    , maskScale : Float
    , loadingMasks : Bool

    -- Search
    , groupSearchQuery : String
    , searchResults : List SearchResult

    -- Project config (fetched from server)
    , projectConfig : Maybe ProjectConfig
    }


type Page
    = SetupPage
    | GroupsPage
    | MasksPage


type alias GroupSummary =
    { groupKey : String
    , groupDisplay : Dict String String
    , count : Int
    }


type alias FrameSummary =
    { pk : String
    , imgPath : String
    , masksCached : Bool
    }


type alias MaskMeta =
    { maskId : Int
    , score : Maybe Float
    , area : Maybe Int
    , url : String
    }


type alias SearchResult =
    { pk : String
    , imgPath : String
    , groupKey : String
    , groupDisplay : Dict String String
    }


type alias ProjectConfig =
    { filterQuery : String
    , groupBy : List String
    , imgPath : String
    , primaryKey : String
    , rootDpath : String
    , sam2Model : String
    , device : String
    , pointsPerCrop : Int
    , cropsNLayers : Int
    , predIouThresh : Float
    , stabilityScoreThresh : Float
    , cropsNmsThresh : Float
    }


init : () -> Url -> Nav.Key -> ( Model, Cmd Msg )
init _ url key =
    let
        route =
            parseUrl url

        initState =
            case route of
                Just (ProjectRoute pid) ->
                    { page = GroupsPage, projectId = Just pid, frameIdx = 0, cmd = Cmd.batch [ fetchGroups pid, fetchProjectConfig pid ] }

                Just (FramesRoute pid idx) ->
                    -- Treat FramesRoute same as MasksRoute (merged pages)
                    { page = MasksPage, projectId = Just pid, frameIdx = idx, cmd = Cmd.batch [ fetchGroups pid, fetchFrames pid ] }

                Just (MasksRoute pid idx) ->
                    { page = MasksPage, projectId = Just pid, frameIdx = idx, cmd = Cmd.batch [ fetchGroups pid, fetchFrames pid ] }

                Just SetupRoute ->
                    { page = SetupPage, projectId = Nothing, frameIdx = 0, cmd = Cmd.none }

                Nothing ->
                    { page = SetupPage, projectId = Nothing, frameIdx = 0, cmd = Cmd.none }
    in
    ( { key = key
      , url = url
      , page = initState.page
      , error = Nothing
      , csvFile = Nothing
      , filterQuery = "SELECT * FROM master_df"
      , groupBy = "Dataset"
      , imgPath = "CONCAT('/local/scratch/datasets/jiggins/butterflies', RIGHT(filepath, LEN(filepath) - 6))"
      , primaryKey = "Image_name"
      , rootDpath = "/local/scratch/stevens.994/datasets/cambridge-segmented"
      , sam2Model = "facebook/sam2.1-hiera-tiny"
      , device = "cuda"
      , pointsPerCrop = "48"
      , cropsNLayers = "0"
      , predIouThresh = "0.88"
      , stabilityScoreThresh = "0.95"
      , cropsNmsThresh = "0.7"
      , projectId = initState.projectId
      , columns = []
      , groupCount = 0
      , rowCount = 0
      , groups = []
      , selectedGroupKey = Nothing
      , nRefFrames = "5"
      , seed = "0"
      , frames = []
      , currentFrameIndex = initState.frameIdx
      , masks = []
      , selectedMasks = Dict.empty
      , maskScale = 0.25
      , loadingMasks = False
      , groupSearchQuery = ""
      , searchResults = []
      , projectConfig = Nothing
      }
    , initState.cmd
    )


type Route
    = SetupRoute
    | ProjectRoute String
    | FramesRoute String Int
    | MasksRoute String Int


parseUrl : Url -> Maybe Route
parseUrl url =
    let
        params =
            Maybe.withDefault "" url.query
                |> String.split "&"
                |> List.filterMap
                    (\s ->
                        case String.split "=" s of
                            [ k, v ] ->
                                Just ( k, Url.percentDecode v |> Maybe.withDefault v )

                            _ ->
                                Nothing
                    )
                |> Dict.fromList

        projectId =
            Dict.get "project" params

        page =
            Dict.get "page" params |> Maybe.withDefault "setup"

        frameIdx =
            Dict.get "frame" params |> Maybe.andThen String.toInt |> Maybe.withDefault 0
    in
    case ( projectId, page ) of
        ( Just pid, "groups" ) ->
            Just (ProjectRoute pid)

        ( Just pid, "frames" ) ->
            -- Treat "frames" as "masks" (merged pages)
            Just (MasksRoute pid frameIdx)

        ( Just pid, "masks" ) ->
            Just (MasksRoute pid frameIdx)

        _ ->
            Just SetupRoute


routeToUrl : Route -> String
routeToUrl route =
    case route of
        SetupRoute ->
            "/"

        ProjectRoute pid ->
            "/?project=" ++ Url.percentEncode pid ++ "&page=groups"

        FramesRoute pid frameIdx ->
            -- Redirect to masks (merged pages)
            "/?project=" ++ Url.percentEncode pid ++ "&page=masks&frame=" ++ String.fromInt frameIdx

        MasksRoute pid frameIdx ->
            "/?project=" ++ Url.percentEncode pid ++ "&page=masks&frame=" ++ String.fromInt frameIdx



-- MSG


type Msg
    = -- URL navigation
      UrlChanged Url
    | LinkClicked Browser.UrlRequest
      -- Setup form
    | CsvSelected File
    | SetFilterQuery String
    | SetGroupBy String
    | SetImgPath String
    | SetPrimaryKey String
    | SetRootDpath String
    | SetSam2Model String
    | SetDevice String
    | SetPointsPerCrop String
    | SetCropsNLayers String
    | SetPredIouThresh String
    | SetStabilityScoreThresh String
    | SetCropsNmsThresh String
    | SubmitProject
    | GotProjectCreated (Result Http.Error ProjectCreatedResponse)
      -- Groups
    | GotGroups (Result Http.Error GroupsResponse)
    | GotProjectInfo (Result Http.Error ProjectCreatedResponse)
    | GotProjectConfig (Result Http.Error ProjectConfig)
    | SelectGroup String
    | SetNRefFrames String
    | SetSeed String
    | SampleFrames
    | GotSampledFrames (Result Http.Error SampledFramesResponse)
      -- Frames
    | GotFrames (Result Http.Error SampledFramesResponse)
    | GoToFrame Int
    | ComputeMasks
    | GotMasks (Result Http.Error MasksResponse)
      -- Mask selection
    | ToggleMask Int
    | SetMaskLabel Int String
    | SaveSelection
    | GotSelectionSaved (Result Http.Error SelectionSavedResponse)
      -- Navigation
    | GoToPage Page
    | DismissError
      -- Search
    | SetGroupSearchQuery String
    | SearchAllFrames
    | GotSearchResults (Result Http.Error SearchResponse)
    | GoToSearchResult SearchResult


type alias ProjectCreatedResponse =
    { projectId : String
    , columns : List String
    , groupCount : Int
    , rowCount : Int
    }


type alias GroupsResponse =
    { groups : List GroupSummary
    , total : Int
    }


type alias SampledFramesResponse =
    { frames : List FrameSummary
    }


type alias MasksResponse =
    { scale : Float
    , masks : List MaskMeta
    }


type alias SelectionSavedResponse =
    { savedFpath : String
    }


type alias SearchResponse =
    { results : List SearchResult
    }



-- UPDATE


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        UrlChanged url ->
            let
                route =
                    parseUrl url
            in
            case route of
                Just (ProjectRoute pid) ->
                    ( { model | url = url, page = GroupsPage, projectId = Just pid }
                    , Cmd.batch [ fetchGroups pid, fetchProjectConfig pid ]
                    )

                Just (FramesRoute pid idx) ->
                    -- Treat FramesRoute same as MasksRoute (merged pages)
                    let
                        cmd =
                            if List.isEmpty model.frames then
                                fetchFrames pid

                            else
                                Cmd.none
                    in
                    ( { model | url = url, page = MasksPage, projectId = Just pid, currentFrameIndex = idx }
                    , cmd
                    )

                Just (MasksRoute pid idx) ->
                    -- Only fetch frames if we don't have them
                    let
                        cmd =
                            if List.isEmpty model.frames then
                                fetchFrames pid

                            else
                                Cmd.none
                    in
                    ( { model | url = url, page = MasksPage, projectId = Just pid, currentFrameIndex = idx }
                    , cmd
                    )

                Just SetupRoute ->
                    ( { model | url = url, page = SetupPage }, Cmd.none )

                Nothing ->
                    ( { model | url = url, page = SetupPage }, Cmd.none )

        LinkClicked urlRequest ->
            case urlRequest of
                Browser.Internal url ->
                    ( model, Nav.pushUrl model.key (Url.toString url) )

                Browser.External href ->
                    ( model, Nav.load href )

        CsvSelected file ->
            ( { model | csvFile = Just file }, Cmd.none )

        SetFilterQuery val ->
            ( { model | filterQuery = val }, Cmd.none )

        SetGroupBy val ->
            ( { model | groupBy = val }, Cmd.none )

        SetImgPath val ->
            ( { model | imgPath = val }, Cmd.none )

        SetPrimaryKey val ->
            ( { model | primaryKey = val }, Cmd.none )

        SetRootDpath val ->
            ( { model | rootDpath = val }, Cmd.none )

        SetSam2Model val ->
            ( { model | sam2Model = val }, Cmd.none )

        SetDevice val ->
            ( { model | device = val }, Cmd.none )

        SetPointsPerCrop val ->
            ( { model | pointsPerCrop = val }, Cmd.none )

        SetCropsNLayers val ->
            ( { model | cropsNLayers = val }, Cmd.none )

        SetPredIouThresh val ->
            ( { model | predIouThresh = val }, Cmd.none )

        SetStabilityScoreThresh val ->
            ( { model | stabilityScoreThresh = val }, Cmd.none )

        SetCropsNmsThresh val ->
            ( { model | cropsNmsThresh = val }, Cmd.none )

        SubmitProject ->
            case model.csvFile of
                Just file ->
                    ( model, createProject model file )

                Nothing ->
                    ( { model | error = Just "Please select a CSV file" }, Cmd.none )

        GotProjectCreated result ->
            case result of
                Ok resp ->
                    ( { model
                        | projectId = Just resp.projectId
                        , columns = resp.columns
                        , groupCount = resp.groupCount
                        , rowCount = resp.rowCount
                        , page = GroupsPage
                        , error = Nothing
                      }
                    , Cmd.batch
                        [ fetchGroups resp.projectId
                        , Nav.pushUrl model.key (routeToUrl (ProjectRoute resp.projectId))
                        ]
                    )

                Err err ->
                    ( { model | error = Just (httpErrorToString err) }, Cmd.none )

        GotGroups result ->
            case result of
                Ok resp ->
                    ( { model | groups = resp.groups, error = Nothing }, Cmd.none )

                Err err ->
                    ( { model | error = Just (httpErrorToString err) }, Cmd.none )

        GotProjectInfo result ->
            case result of
                Ok resp ->
                    ( { model
                        | columns = resp.columns
                        , groupCount = resp.groupCount
                        , rowCount = resp.rowCount
                        , error = Nothing
                      }
                    , Cmd.none
                    )

                Err err ->
                    ( { model | error = Just (httpErrorToString err) }, Cmd.none )

        GotProjectConfig result ->
            case result of
                Ok config ->
                    ( { model | projectConfig = Just config, error = Nothing }, Cmd.none )

                Err err ->
                    ( { model | error = Just (httpErrorToString err) }, Cmd.none )

        SelectGroup groupKey ->
            ( { model | selectedGroupKey = Just groupKey }, Cmd.none )

        SetNRefFrames val ->
            ( { model | nRefFrames = val }, Cmd.none )

        SetSeed val ->
            ( { model | seed = val }, Cmd.none )

        SampleFrames ->
            case ( model.projectId, model.selectedGroupKey ) of
                ( Just pid, Just gk ) ->
                    ( model, sampleFrames pid gk model.nRefFrames model.seed )

                _ ->
                    ( { model | error = Just "Select a group first" }, Cmd.none )

        GotSampledFrames result ->
            case result of
                Ok resp ->
                    case model.projectId of
                        Just pid ->
                            -- Go directly to MasksPage
                            ( { model
                                | frames = resp.frames
                                , currentFrameIndex = 0
                                , page = MasksPage
                                , error = Nothing
                                , masks = []
                                , selectedMasks = Dict.empty
                                , loadingMasks = True
                              }
                            , Cmd.batch
                                [ Nav.pushUrl model.key (routeToUrl (MasksRoute pid 0))
                                , case List.head resp.frames of
                                    Just frame ->
                                        computeMasks pid frame.pk model.maskScale

                                    Nothing ->
                                        Cmd.none
                                ]
                            )

                        Nothing ->
                            ( { model | error = Just "No project ID" }, Cmd.none )

                Err err ->
                    ( { model | error = Just (httpErrorToString err) }, Cmd.none )

        GotFrames result ->
            case result of
                Ok resp ->
                    -- Just update frames, never auto-compute masks (user clicks Compute Masks button)
                    ( { model | frames = resp.frames, error = Nothing }, Cmd.none )

                Err err ->
                    ( { model | error = Just (httpErrorToString err) }, Cmd.none )

        GoToFrame idx ->
            if idx >= 0 && idx < List.length model.frames then
                -- Navigate to frame and auto-compute masks
                case model.projectId of
                    Just pid ->
                        ( { model
                            | currentFrameIndex = idx
                            , masks = []
                            , selectedMasks = Dict.empty
                            , loadingMasks = True
                          }
                        , Cmd.batch
                            [ Nav.pushUrl model.key (routeToUrl (MasksRoute pid idx))
                            , case List.head (List.drop idx model.frames) of
                                Just frame ->
                                    computeMasks pid frame.pk model.maskScale

                                Nothing ->
                                    Cmd.none
                            ]
                        )

                    Nothing ->
                        ( model, Cmd.none )

            else
                ( model, Cmd.none )

        ComputeMasks ->
            case model.projectId of
                Just pid ->
                    case List.head (List.drop model.currentFrameIndex model.frames) of
                        Just frame ->
                            -- Just compute masks, no URL change (page state is internal)
                            ( { model | loadingMasks = True, page = MasksPage }
                            , computeMasks pid frame.pk model.maskScale
                            )

                        Nothing ->
                            ( model, Cmd.none )

                Nothing ->
                    ( model, Cmd.none )

        GotMasks result ->
            case result of
                Ok resp ->
                    -- Just update masks, URL was already pushed from ComputeMasks or init
                    ( { model
                        | masks = resp.masks
                        , loadingMasks = False
                        , page = MasksPage
                        , error = Nothing
                      }
                    , Cmd.none
                    )

                Err err ->
                    ( { model | error = Just (httpErrorToString err), loadingMasks = False }, Cmd.none )

        ToggleMask maskId ->
            let
                newSelected =
                    if Dict.member maskId model.selectedMasks then
                        Dict.remove maskId model.selectedMasks

                    else
                        -- Auto-assign next label (A=0, B=1, C=2, etc.)
                        Dict.insert maskId (Dict.size model.selectedMasks) model.selectedMasks
            in
            ( { model | selectedMasks = newSelected }, Cmd.none )

        SetMaskLabel maskId labelStr ->
            case letterToIndex labelStr of
                Just label ->
                    ( { model | selectedMasks = Dict.insert maskId label model.selectedMasks }, Cmd.none )

                Nothing ->
                    ( model, Cmd.none )

        SaveSelection ->
            case model.projectId of
                Just pid ->
                    case List.head (List.drop model.currentFrameIndex model.frames) of
                        Just frame ->
                            ( model, saveSelection pid frame.pk model.selectedMasks )

                        Nothing ->
                            ( model, Cmd.none )

                Nothing ->
                    ( model, Cmd.none )

        GotSelectionSaved result ->
            case result of
                Ok _ ->
                    let
                        nextIndex =
                            model.currentFrameIndex + 1
                    in
                    case model.projectId of
                        Just pid ->
                            if nextIndex < List.length model.frames then
                                -- Go to next frame and auto-compute masks
                                ( { model
                                    | currentFrameIndex = nextIndex
                                    , masks = []
                                    , selectedMasks = Dict.empty
                                    , page = MasksPage
                                    , error = Nothing
                                    , loadingMasks = True
                                  }
                                , Cmd.batch
                                    [ Nav.pushUrl model.key (routeToUrl (MasksRoute pid nextIndex))
                                    , case List.head (List.drop nextIndex model.frames) of
                                        Just frame ->
                                            computeMasks pid frame.pk model.maskScale

                                        Nothing ->
                                            Cmd.none
                                    ]
                                )

                            else
                                -- All frames done, go back to groups
                                ( { model
                                    | error = Nothing
                                    , page = GroupsPage
                                    , masks = []
                                    , selectedMasks = Dict.empty
                                  }
                                , Nav.pushUrl model.key (routeToUrl (ProjectRoute pid))
                                )

                        Nothing ->
                            ( model, Cmd.none )

                Err err ->
                    ( { model | error = Just (httpErrorToString err) }, Cmd.none )

        GoToPage page ->
            case ( page, model.projectId ) of
                ( GroupsPage, Just pid ) ->
                    ( { model | page = page }, Nav.pushUrl model.key (routeToUrl (ProjectRoute pid)) )

                ( MasksPage, Just pid ) ->
                    ( { model | page = page }, Nav.pushUrl model.key (routeToUrl (MasksRoute pid model.currentFrameIndex)) )

                _ ->
                    ( { model | page = page }, Cmd.none )

        DismissError ->
            ( { model | error = Nothing }, Cmd.none )

        SetGroupSearchQuery query ->
            ( { model | groupSearchQuery = query }, Cmd.none )

        SearchAllFrames ->
            case model.projectId of
                Just pid ->
                    if String.length model.groupSearchQuery >= 2 then
                        ( model, searchAllFrames pid model.groupSearchQuery )

                    else
                        ( { model | searchResults = [] }, Cmd.none )

                Nothing ->
                    ( model, Cmd.none )

        GotSearchResults result ->
            case result of
                Ok resp ->
                    ( { model | searchResults = resp.results }, Cmd.none )

                Err err ->
                    ( { model | error = Just (httpErrorToString err) }, Cmd.none )

        GoToSearchResult searchResult ->
            case model.projectId of
                Just pid ->
                    -- Add the frame to sampled_frames and go to MasksPage
                    let
                        newFrame =
                            { pk = searchResult.pk
                            , imgPath = searchResult.imgPath
                            , masksCached = False
                            }
                    in
                    ( { model
                        | frames = [ newFrame ]
                        , currentFrameIndex = 0
                        , page = MasksPage
                        , masks = []
                        , selectedMasks = Dict.empty
                        , loadingMasks = True
                        , searchResults = []
                        , groupSearchQuery = ""
                      }
                    , Cmd.batch
                        [ Nav.pushUrl model.key (routeToUrl (MasksRoute pid 0))
                        , computeMasks pid searchResult.pk model.maskScale
                        ]
                    )

                Nothing ->
                    ( model, Cmd.none )



-- HTTP


createProject : Model -> File -> Cmd Msg
createProject model file =
    Http.request
        { method = "POST"
        , headers = []
        , url = "/api/projects"
        , body =
            Http.multipartBody
                [ Http.filePart "csv" file
                , Http.stringPart "filter_query" model.filterQuery
                , Http.stringPart "group_by" model.groupBy
                , Http.stringPart "img_path" model.imgPath
                , Http.stringPart "primary_key" model.primaryKey
                , Http.stringPart "root_dpath" model.rootDpath
                , Http.stringPart "sam2_model" model.sam2Model
                , Http.stringPart "device" model.device
                , Http.stringPart "points_per_crop" model.pointsPerCrop
                , Http.stringPart "crops_n_layers" model.cropsNLayers
                , Http.stringPart "pred_iou_thresh" model.predIouThresh
                , Http.stringPart "stability_score_thresh" model.stabilityScoreThresh
                , Http.stringPart "crops_nms_thresh" model.cropsNmsThresh
                ]
        , expect = Http.expectJson GotProjectCreated projectCreatedDecoder
        , timeout = Nothing
        , tracker = Nothing
        }


fetchGroups : String -> Cmd Msg
fetchGroups projectId =
    Http.get
        { url = "/api/projects/" ++ projectId ++ "/groups?limit=1000"
        , expect = Http.expectJson GotGroups groupsDecoder
        }


fetchProjectConfig : String -> Cmd Msg
fetchProjectConfig projectId =
    Http.get
        { url = "/api/projects/" ++ projectId ++ "/spec"
        , expect = Http.expectJson GotProjectConfig projectConfigDecoder
        }


fetchFrames : String -> Cmd Msg
fetchFrames projectId =
    Http.get
        { url = "/api/projects/" ++ projectId ++ "/frames"
        , expect = Http.expectJson GotFrames sampledFramesDecoder
        }


sampleFrames : String -> String -> String -> String -> Cmd Msg
sampleFrames projectId groupKey nRefFrames seed =
    Http.request
        { method = "POST"
        , headers = []
        , url = "/api/projects/" ++ projectId ++ "/groups/" ++ groupKey ++ "/sample"
        , body =
            Http.jsonBody
                (E.object
                    [ ( "n_ref_frames", E.int (Maybe.withDefault 5 (String.toInt nRefFrames)) )
                    , ( "seed", E.int (Maybe.withDefault 0 (String.toInt seed)) )
                    ]
                )
        , expect = Http.expectJson GotSampledFrames sampledFramesDecoder
        , timeout = Nothing
        , tracker = Nothing
        }


computeMasks : String -> String -> Float -> Cmd Msg
computeMasks projectId pk scale =
    Http.request
        { method = "POST"
        , headers = []
        , url = "/api/projects/" ++ projectId ++ "/frames/" ++ pk ++ "/masks"
        , body =
            Http.jsonBody
                (E.object
                    [ ( "scale", E.float scale )
                    ]
                )
        , expect = Http.expectJson GotMasks masksDecoder
        , timeout = Just 120000
        , tracker = Nothing
        }


saveSelection : String -> String -> Dict Int Int -> Cmd Msg
saveSelection projectId pk selectedMasks =
    let
        items =
            Dict.toList selectedMasks

        maskIds =
            List.map Tuple.first items

        -- Add 1 to labels since backend uses 0 as background
        labels =
            List.map (\( _, label ) -> label + 1) items
    in
    Http.request
        { method = "POST"
        , headers = []
        , url = "/api/projects/" ++ projectId ++ "/frames/" ++ pk ++ "/selection"
        , body =
            Http.jsonBody
                (E.object
                    [ ( "mask_ids", E.list E.int maskIds )
                    , ( "labels", E.list E.int labels )
                    ]
                )
        , expect = Http.expectJson GotSelectionSaved selectionSavedDecoder
        , timeout = Nothing
        , tracker = Nothing
        }


searchAllFrames : String -> String -> Cmd Msg
searchAllFrames projectId query =
    Http.get
        { url = "/api/projects/" ++ projectId ++ "/search?q=" ++ Url.percentEncode query
        , expect = Http.expectJson GotSearchResults searchResponseDecoder
        }



-- DECODERS


projectCreatedDecoder : D.Decoder ProjectCreatedResponse
projectCreatedDecoder =
    D.map4 ProjectCreatedResponse
        (D.field "project_id" D.string)
        (D.field "columns" (D.list D.string))
        (D.field "group_count" D.int)
        (D.field "row_count" D.int)


groupsDecoder : D.Decoder GroupsResponse
groupsDecoder =
    D.map2 GroupsResponse
        (D.field "groups" (D.list groupSummaryDecoder))
        (D.field "total" D.int)


groupSummaryDecoder : D.Decoder GroupSummary
groupSummaryDecoder =
    D.map3 GroupSummary
        (D.field "group_key" D.string)
        (D.field "group_display" (D.dict D.string))
        (D.field "count" D.int)


sampledFramesDecoder : D.Decoder SampledFramesResponse
sampledFramesDecoder =
    D.map SampledFramesResponse
        (D.field "frames" (D.list frameSummaryDecoder))


frameSummaryDecoder : D.Decoder FrameSummary
frameSummaryDecoder =
    D.map3 FrameSummary
        (D.field "pk" D.string)
        (D.field "img_path" D.string)
        (D.oneOf [ D.field "masks_cached" D.bool, D.succeed False ])


masksDecoder : D.Decoder MasksResponse
masksDecoder =
    D.map2 MasksResponse
        (D.field "scale" D.float)
        (D.field "masks" (D.list maskMetaDecoder))


maskMetaDecoder : D.Decoder MaskMeta
maskMetaDecoder =
    D.map4 MaskMeta
        (D.field "mask_id" D.int)
        (D.maybe (D.field "score" D.float))
        (D.maybe (D.field "area" D.int))
        (D.field "url" D.string)


selectionSavedDecoder : D.Decoder SelectionSavedResponse
selectionSavedDecoder =
    D.map SelectionSavedResponse
        (D.field "saved_fpath" D.string)


searchResponseDecoder : D.Decoder SearchResponse
searchResponseDecoder =
    D.map SearchResponse
        (D.field "results" (D.list searchResultDecoder))


searchResultDecoder : D.Decoder SearchResult
searchResultDecoder =
    D.map4 SearchResult
        (D.field "pk" D.string)
        (D.field "img_path" D.string)
        (D.field "group_key" D.string)
        (D.field "group_display" (D.dict D.string))


projectConfigDecoder : D.Decoder ProjectConfig
projectConfigDecoder =
    D.succeed ProjectConfig
        |> andMap (D.field "filter_query" D.string)
        |> andMap (D.field "group_by" (D.list D.string))
        |> andMap (D.field "img_path" D.string)
        |> andMap (D.field "primary_key" D.string)
        |> andMap (D.field "root_dpath" D.string)
        |> andMap (D.field "sam2_model" D.string)
        |> andMap (D.field "device" D.string)
        |> andMap (D.field "points_per_crop" D.int)
        |> andMap (D.field "crops_n_layers" D.int)
        |> andMap (D.field "pred_iou_thresh" D.float)
        |> andMap (D.field "stability_score_thresh" D.float)
        |> andMap (D.field "crops_nms_thresh" D.float)


andMap : D.Decoder a -> D.Decoder (a -> b) -> D.Decoder b
andMap =
    D.map2 (|>)


httpErrorToString : Http.Error -> String
httpErrorToString err =
    case err of
        Http.BadUrl url ->
            "Bad URL: " ++ url

        Http.Timeout ->
            "Request timed out"

        Http.NetworkError ->
            "Network error"

        Http.BadStatus status ->
            "Bad status: " ++ String.fromInt status

        Http.BadBody body ->
            "Bad body: " ++ body



-- VIEW


view : Model -> Browser.Document Msg
view model =
    { title = "SST GUI"
    , body =
        [ div [ class "max-w-6xl mx-auto p-4" ]
            [ viewError model.error
            , case model.page of
                SetupPage ->
                    viewSetupPage model

                GroupsPage ->
                    viewGroupsPage model

                MasksPage ->
                    viewMasksPage model
            ]
        ]
    }


viewError : Maybe String -> Html Msg
viewError maybeError =
    case maybeError of
        Just err ->
            div [ class "bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4 flex justify-between items-center" ]
                [ span [] [ text err ]
                , button [ onClick DismissError, class "text-red-700 font-bold" ] [ text "x" ]
                ]

        Nothing ->
            text ""


viewSetupPage : Model -> Html Msg
viewSetupPage model =
    div []
        [ h2 [ class "text-2xl font-bold mb-4" ] [ text "Project Setup" ]

        -- CSV File (full width)
        , div [ class "mb-4" ]
            [ label [ class "block text-sm font-medium mb-1" ] [ text "CSV File" ]
            , p [ class "text-xs text-gray-500 mb-1" ] [ text "Upload a CSV file containing frame metadata" ]
            , input
                [ type_ "file"
                , accept ".csv"
                , on "change" (D.map CsvSelected fileDecoder)
                , class "block w-full text-sm border rounded p-2"
                ]
                []
            ]

        -- Two-column grid for the rest
        , div [ class "grid grid-cols-1 md:grid-cols-2 gap-4" ]
            [ viewInputWithDesc "Filter Query (SQL)" "SQL query to filter the CSV data" model.filterQuery SetFilterQuery
            , viewInputWithDesc "Group By (columns)" "Columns used to group frames for batch processing" model.groupBy SetGroupBy
            , viewInputWithDesc "Image Path (expression)" "Column name or SQL expression for image file paths" model.imgPath SetImgPath
            , viewInputWithDesc "Primary Key (column)" "Unique identifier column for each frame" model.primaryKey SetPrimaryKey
            , viewInputWithDesc "Root Directory" "Root directory for saving output masks" model.rootDpath SetRootDpath
            , div []
                [ label [ class "block text-sm font-medium mb-1" ] [ text "SAM2 Model" ]
                , p [ class "text-xs text-gray-500 mb-1" ] [ text "Model variant. Larger = better quality but slower" ]
                , select
                    [ onInput SetSam2Model
                    , value model.sam2Model
                    , class "block w-full border rounded p-2"
                    ]
                    [ option [ value "facebook/sam2.1-hiera-tiny" ] [ text "sam2.1-hiera-tiny" ]
                    , option [ value "facebook/sam2.1-hiera-small" ] [ text "sam2.1-hiera-small" ]
                    , option [ value "facebook/sam2.1-hiera-base-plus" ] [ text "sam2.1-hiera-base-plus" ]
                    , option [ value "facebook/sam2.1-hiera-large" ] [ text "sam2.1-hiera-large" ]
                    ]
                ]
            , div []
                [ label [ class "block text-sm font-medium mb-1" ] [ text "Device" ]
                , p [ class "text-xs text-gray-500 mb-1" ] [ text "Hardware device for inference (cuda/cpu)" ]
                , select
                    [ onInput SetDevice
                    , value model.device
                    , class "block w-full border rounded p-2"
                    ]
                    [ option [ value "cuda" ] [ text "cuda" ]
                    , option [ value "cpu" ] [ text "cpu" ]
                    ]
                ]
            , div []
                [ label [ class "block text-sm font-medium mb-1" ] [ text "Points Per Crop" ]
                , p [ class "text-xs text-gray-500 mb-1" ] [ text "Point prompts per crop. Higher = more masks" ]
                , select
                    [ onInput SetPointsPerCrop
                    , value model.pointsPerCrop
                    , class "block w-full border rounded p-2"
                    ]
                    [ option [ value "32" ] [ text "32 (fewer masks, faster)" ]
                    , option [ value "48" ] [ text "48 (default)" ]
                    , option [ value "64" ] [ text "64 (more masks)" ]
                    , option [ value "96" ] [ text "96 (many masks, slower)" ]
                    ]
                ]
            , div []
                [ label [ class "block text-sm font-medium mb-1" ] [ text "Crop Layers" ]
                , p [ class "text-xs text-gray-500 mb-1" ] [ text "Multi-scale detection (buggy, keep at 0)" ]
                , select
                    [ onInput SetCropsNLayers
                    , value model.cropsNLayers
                    , class "block w-full border rounded p-2"
                    ]
                    [ option [ value "0" ] [ text "0 (disabled)" ]
                    , option [ value "1" ] [ text "1 (4 crops)" ]
                    , option [ value "2" ] [ text "2 (16 crops)" ]
                    ]
                ]
            , div []
                [ label [ class "block text-sm font-medium mb-1" ] [ text "Predicted IoU Threshold" ]
                , p [ class "text-xs text-gray-500 mb-1" ] [ text "Model confidence filter. Lower = keep more" ]
                , select
                    [ onInput SetPredIouThresh
                    , value model.predIouThresh
                    , class "block w-full border rounded p-2"
                    ]
                    [ option [ value "0.5" ] [ text "0.5 (keep most)" ]
                    , option [ value "0.7" ] [ text "0.7 (lenient)" ]
                    , option [ value "0.88" ] [ text "0.88 (default)" ]
                    , option [ value "0.95" ] [ text "0.95 (strict)" ]
                    ]
                ]
            , div []
                [ label [ class "block text-sm font-medium mb-1" ] [ text "Stability Threshold" ]
                , p [ class "text-xs text-gray-500 mb-1" ] [ text "Mask consistency filter. Lower = keep more" ]
                , select
                    [ onInput SetStabilityScoreThresh
                    , value model.stabilityScoreThresh
                    , class "block w-full border rounded p-2"
                    ]
                    [ option [ value "0.5" ] [ text "0.5 (keep most)" ]
                    , option [ value "0.75" ] [ text "0.75 (lenient)" ]
                    , option [ value "0.85" ] [ text "0.85 (moderate)" ]
                    , option [ value "0.95" ] [ text "0.95 (default)" ]
                    ]
                ]
            , div []
                [ label [ class "block text-sm font-medium mb-1" ] [ text "Crop NMS Threshold" ]
                , p [ class "text-xs text-gray-500 mb-1" ] [ text "Duplicate removal. Lower = fewer dupes" ]
                , select
                    [ onInput SetCropsNmsThresh
                    , value model.cropsNmsThresh
                    , class "block w-full border rounded p-2"
                    ]
                    [ option [ value "0.5" ] [ text "0.5 (aggressive)" ]
                    , option [ value "0.7" ] [ text "0.7 (default)" ]
                    , option [ value "0.9" ] [ text "0.9 (keep overlapping)" ]
                    , option [ value "1.0" ] [ text "1.0 (no dedup)" ]
                    ]
                ]
            ]

        -- Button (full width)
        , div [ class "mt-4" ]
            [ button
                [ onClick SubmitProject
                , class "bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
                ]
                [ text "Create Project" ]
            ]
        ]


viewInput : String -> String -> (String -> Msg) -> Html Msg
viewInput labelText val toMsg =
    div []
        [ label [ class "block text-sm font-medium mb-1" ] [ text labelText ]
        , input
            [ type_ "text"
            , value val
            , onInput toMsg
            , class "block w-full border rounded p-2"
            ]
            []
        ]


viewInputWithDesc : String -> String -> String -> (String -> Msg) -> Html Msg
viewInputWithDesc labelText description val toMsg =
    div []
        [ label [ class "block text-sm font-medium mb-1" ] [ text labelText ]
        , p [ class "text-xs text-gray-500 mb-1" ] [ text description ]
        , input
            [ type_ "text"
            , value val
            , onInput toMsg
            , class "block w-full border rounded p-2"
            ]
            []
        ]


viewProjectConfig : ProjectConfig -> Html Msg
viewProjectConfig config =
    div [ class "mb-4 p-3 bg-blue-50 border border-blue-200 rounded text-sm" ]
        [ h3 [ class "font-semibold mb-2" ] [ text "Project Configuration" ]
        , div [ class "grid grid-cols-1 md:grid-cols-2 gap-2" ]
            [ viewConfigItem "filter_query" config.filterQuery "SQL query to filter the CSV data"
            , viewConfigItem "group_by" (String.join ", " config.groupBy) "Columns used to group frames"
            , viewConfigItem "img_path" config.imgPath "Column or expression for image file paths"
            , viewConfigItem "primary_key" config.primaryKey "Unique identifier column for each frame"
            , viewConfigItem "root_dpath" config.rootDpath "Root directory for saving output masks"
            , viewConfigItem "sam2_model" config.sam2Model "SAM2 model variant for mask generation"
            , viewConfigItem "device" config.device "Hardware device for inference (cuda/cpu)"
            , viewConfigItem "points_per_crop" (String.fromInt config.pointsPerCrop) "Number of point prompts per crop. Higher = more masks"
            , viewConfigItem "crops_n_layers" (String.fromInt config.cropsNLayers) "Multi-scale detection. Crops image into regions for smaller objects"
            , viewConfigItem "pred_iou_thresh" (String.fromFloat config.predIouThresh) "Model's confidence in mask accuracy. Lower = keep less confident masks"
            , viewConfigItem "stability_score_thresh" (String.fromFloat config.stabilityScoreThresh) "Mask consistency under threshold changes. Lower = keep fragile masks"
            , viewConfigItem "crops_nms_thresh" (String.fromFloat config.cropsNmsThresh) "Removes duplicate masks from overlapping crops"
            ]
        ]


viewConfigItem : String -> String -> String -> Html Msg
viewConfigItem label val description =
    div [ class "bg-white p-2 rounded border" ]
        [ div [ class "font-medium text-gray-700" ] [ text label ]
        , div [ class "text-blue-600" ] [ text val ]
        , if String.isEmpty description then
            text ""

          else
            div [ class "text-xs text-gray-500 mt-1" ] [ text description ]
        ]


viewGroupsPage : Model -> Html Msg
viewGroupsPage model =
    div []
        [ h2 [ class "text-2xl font-bold mb-4" ] [ text "Select Group" ]
        , div [ class "mb-4 text-sm text-gray-600" ]
            [ text ("Project: " ++ Maybe.withDefault "" model.projectId)
            , text (" | " ++ String.fromInt model.rowCount ++ " rows, " ++ String.fromInt model.groupCount ++ " groups")
            ]

        -- Search box
        , div [ class "mb-4 p-3 bg-gray-50 border rounded" ]
            [ div [ class "flex gap-2 items-center" ]
                [ span [ class "text-sm font-medium" ] [ text "Jump to frame:" ]
                , input
                    [ type_ "text"
                    , placeholder "Search by name (min 2 chars)..."
                    , value model.groupSearchQuery
                    , onInput SetGroupSearchQuery
                    , onEnter SearchAllFrames
                    , class "border rounded px-2 py-1 text-sm flex-1"
                    ]
                    []
                , button
                    [ onClick SearchAllFrames
                    , class "bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600 text-sm"
                    ]
                    [ text "Search" ]
                ]
            , if not (List.isEmpty model.searchResults) then
                div [ class "mt-2 border rounded bg-white max-h-48 overflow-y-auto" ]
                    (List.map viewSearchResult model.searchResults)

              else if String.length model.groupSearchQuery >= 2 then
                div [ class "mt-2 text-sm text-gray-500" ] [ text "No results found" ]

              else
                text ""
            ]

        -- Sampling options
        , div [ class "grid grid-cols-2 gap-4 mb-4" ]
            [ viewInput "Number of reference frames" model.nRefFrames SetNRefFrames
            , viewInput "Random seed" model.seed SetSeed
            ]
        , div [ class "border rounded max-h-96 overflow-y-auto" ]
            (List.map (viewGroupRow model.selectedGroupKey) model.groups)
        , div [ class "mt-4" ]
            [ button
                [ onClick SampleFrames
                , disabled (model.selectedGroupKey == Nothing)
                , class "bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-300"
                ]
                [ text "Sample Frames" ]
            ]

        -- Project config display (at bottom)
        , case model.projectConfig of
            Just config ->
                viewProjectConfig config

            Nothing ->
                text ""
        ]


viewSearchResult : SearchResult -> Html Msg
viewSearchResult result =
    let
        groupText =
            result.groupDisplay
                |> Dict.toList
                |> List.map (\( k, v ) -> k ++ "=" ++ v)
                |> String.join ", "
    in
    div
        [ onClick (GoToSearchResult result)
        , class "p-2 border-b cursor-pointer hover:bg-blue-50 text-sm"
        ]
        [ div [ class "font-medium" ] [ text result.pk ]
        , div [ class "text-gray-500 text-xs" ] [ text groupText ]
        ]


viewGroupRow : Maybe String -> GroupSummary -> Html Msg
viewGroupRow selectedKey group =
    let
        isSelected =
            selectedKey == Just group.groupKey

        displayText =
            group.groupDisplay
                |> Dict.toList
                |> List.map (\( k, v ) -> k ++ "=" ++ v)
                |> String.join ", "
    in
    div
        [ onClick (SelectGroup group.groupKey)
        , class
            ("p-3 border-b cursor-pointer hover:bg-gray-100 "
                ++ (if isSelected then
                        "bg-blue-100"

                    else
                        ""
                   )
            )
        ]
        [ span [ class "font-medium" ] [ text displayText ]
        , span [ class "text-gray-500 ml-2" ] [ text ("(" ++ String.fromInt group.count ++ " frames)") ]
        ]


viewFramesPage : Model -> Html Msg
viewFramesPage model =
    let
        currentFrame =
            List.head (List.drop model.currentFrameIndex model.frames)
    in
    div []
        [ h2 [ class "text-2xl font-bold mb-4" ] [ text "Frames" ]
        , div [ class "mb-4 text-sm text-gray-600" ]
            [ text ("Frame " ++ String.fromInt (model.currentFrameIndex + 1) ++ " of " ++ String.fromInt (List.length model.frames)) ]
        , case currentFrame of
            Just frame ->
                div []
                    [ div [ class "mb-4" ]
                        [ case model.projectId of
                            Just pid ->
                                img
                                    [ src ("/api/projects/" ++ pid ++ "/frames/" ++ frame.pk ++ "/image?scale=" ++ String.fromFloat model.maskScale)
                                    , class "max-w-full border rounded"
                                    ]
                                    []

                            Nothing ->
                                text ""
                        ]
                    , div [ class "text-sm text-gray-600 mb-4" ] [ text ("PK: " ++ frame.pk) ]
                    , div [ class "flex gap-2" ]
                        [ button
                            [ onClick (GoToFrame (model.currentFrameIndex - 1))
                            , disabled (model.currentFrameIndex == 0)
                            , class "bg-gray-200 px-4 py-2 rounded hover:bg-gray-300 disabled:bg-gray-100"
                            ]
                            [ text "Previous" ]
                        , button
                            [ onClick ComputeMasks
                            , disabled model.loadingMasks
                            , class "bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-300"
                            ]
                            [ text
                                (if model.loadingMasks then
                                    "Computing..."

                                 else
                                    "Compute Masks"
                                )
                            ]
                        , button
                            [ onClick (GoToFrame (model.currentFrameIndex + 1))
                            , disabled (model.currentFrameIndex >= List.length model.frames - 1)
                            , class "bg-gray-200 px-4 py-2 rounded hover:bg-gray-300 disabled:bg-gray-100"
                            ]
                            [ text "Next" ]
                        ]
                    ]

            Nothing ->
                text "No frames"
        , div [ class "mt-4" ]
            [ button
                [ onClick (GoToPage GroupsPage)
                , class "text-blue-500 hover:underline"
                ]
                [ text "Back to groups" ]
            ]
        ]


viewMasksPage : Model -> Html Msg
viewMasksPage model =
    let
        currentFrame =
            List.head (List.drop model.currentFrameIndex model.frames)

        selectedMaskIds =
            Dict.keys model.selectedMasks

        hasPrev =
            model.currentFrameIndex > 0

        hasNext =
            model.currentFrameIndex < List.length model.frames - 1
    in
    div []
        [ h2 [ class "text-2xl font-bold mb-4" ] [ text "Select Masks" ]
        , case currentFrame of
            Just frame ->
                div []
                    [ -- Frame info and navigation
                      div [ class "mb-4 flex items-center justify-between" ]
                        [ div [ class "text-sm text-gray-600" ]
                            [ text ("Frame " ++ String.fromInt (model.currentFrameIndex + 1) ++ " of " ++ String.fromInt (List.length model.frames))
                            , text (" | PK: " ++ frame.pk)
                            , text (" | " ++ String.fromInt (List.length model.masks) ++ " masks")
                            ]
                        , div [ class "flex gap-2" ]
                            [ button
                                [ onClick (GoToFrame (model.currentFrameIndex - 1))
                                , disabled (not hasPrev)
                                , class "bg-gray-200 px-3 py-1 rounded hover:bg-gray-300 disabled:opacity-50"
                                ]
                                [ text "← Prev" ]
                            , button
                                [ onClick (GoToFrame (model.currentFrameIndex + 1))
                                , disabled (not hasNext)
                                , class "bg-gray-200 px-3 py-1 rounded hover:bg-gray-300 disabled:opacity-50"
                                ]
                                [ text "Next →" ]
                            ]
                        ]
                    , if model.loadingMasks then
                        div [ class "text-center py-8" ]
                            [ text "Computing masks..." ]

                      else if List.isEmpty model.masks then
                        div [ class "text-center py-8" ]
                            [ button
                                [ onClick ComputeMasks
                                , class "bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
                                ]
                                [ text "Compute Masks" ]
                            ]

                      else
                        div [ class "flex gap-4" ]
                            [ -- Main image with overlays
                              div [ class "flex-1 min-w-0" ]
                                [ case model.projectId of
                                    Just pid ->
                                        div [ class "relative inline-block max-w-full" ]
                                            (img
                                                [ src ("/api/projects/" ++ pid ++ "/frames/" ++ frame.pk ++ "/image?scale=" ++ String.fromFloat model.maskScale)
                                                , class "block border rounded max-w-full"
                                                , id "base-image"
                                                ]
                                                []
                                                :: List.filterMap
                                                    (\maskId ->
                                                        case List.filter (\m -> m.maskId == maskId) model.masks |> List.head of
                                                            Just mask ->
                                                                let
                                                                    label =
                                                                        Dict.get maskId model.selectedMasks |> Maybe.withDefault 0
                                                                in
                                                                Just
                                                                    (div
                                                                        [ class "absolute top-0 left-0 w-full h-full pointer-events-none"
                                                                        , style "background-color" (maskColor (label + 1))
                                                                        , style "opacity" "0.5"
                                                                        , style "-webkit-mask-image" ("url(" ++ mask.url ++ ")")
                                                                        , style "mask-image" ("url(" ++ mask.url ++ ")")
                                                                        , style "-webkit-mask-mode" "luminance"
                                                                        , style "mask-mode" "luminance"
                                                                        , style "-webkit-mask-size" "contain"
                                                                        , style "mask-size" "contain"
                                                                        , style "-webkit-mask-repeat" "no-repeat"
                                                                        , style "mask-repeat" "no-repeat"
                                                                        , style "-webkit-mask-position" "center"
                                                                        , style "mask-position" "center"
                                                                        ]
                                                                        []
                                                                    )

                                                            Nothing ->
                                                                Nothing
                                                    )
                                                    selectedMaskIds
                                            )

                                    Nothing ->
                                        text ""
                                ]

                            -- Mask grid
                            , div [ class "flex-1 min-w-0" ]
                                [ div [ class "grid grid-cols-4 gap-2 max-h-96 overflow-y-auto p-2 border rounded bg-gray-50" ]
                                    (List.indexedMap (viewMaskThumbnail model) model.masks)
                                ]
                            ]
                    , div [ class "mt-4 flex gap-2" ]
                        [ button
                            [ onClick (GoToPage GroupsPage)
                            , class "bg-gray-200 px-4 py-2 rounded hover:bg-gray-300"
                            ]
                            [ text "Back to Groups" ]
                        , button
                            [ onClick SaveSelection
                            , disabled (Dict.isEmpty model.selectedMasks)
                            , class "bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 disabled:bg-gray-300"
                            ]
                            [ text "Save & Continue" ]
                        ]
                    ]

            Nothing ->
                text "No frame selected"
        ]



-- Convert mask index to letter (0→A, 1→B, ... 25→Z, 26→AA, 27→AB, ...)


indexToLetter : Int -> String
indexToLetter idx =
    if idx < 26 then
        String.fromChar (Char.fromCode (65 + idx))

    else
        indexToLetter (idx // 26 - 1) ++ String.fromChar (Char.fromCode (65 + modBy 26 idx))



-- Convert letter back to index (A→0, B→1, ... Z→25, AA→26, AB→27, ...)


letterToIndex : String -> Maybe Int
letterToIndex str =
    let
        upper =
            String.toUpper str

        chars =
            String.toList upper

        charToVal c =
            Char.toCode c - 65
    in
    if List.all (\c -> c >= 'A' && c <= 'Z') chars && not (List.isEmpty chars) then
        Just
            (List.foldl
                (\c acc -> acc * 26 + charToVal c + 1)
                0
                chars
                - 1
            )

    else
        Nothing


maskColor : Int -> String
maskColor maskId =
    let
        -- RGB colors matching mask_browser.py:
        -- 1=red, 2=green, 3=blue, 4=yellow, 5=magenta, 6=cyan, 7=orange, 8=purple, 9=spring green
        colors =
            [ "rgb(255, 0, 0)" -- 1: red
            , "rgb(0, 255, 0)" -- 2: green
            , "rgb(0, 0, 255)" -- 3: blue
            , "rgb(255, 255, 0)" -- 4: yellow
            , "rgb(255, 0, 255)" -- 5: magenta
            , "rgb(0, 255, 255)" -- 6: cyan
            , "rgb(255, 128, 0)" -- 7: orange
            , "rgb(128, 0, 255)" -- 8: purple
            , "rgb(0, 255, 128)" -- 9: spring green
            ]

        colorValue =
            Maybe.withDefault "rgb(255, 0, 0)" (List.head (List.drop (modBy 9 (maskId - 1)) colors))
    in
    colorValue


viewMaskThumbnail : Model -> Int -> MaskMeta -> Html Msg
viewMaskThumbnail model idx mask =
    let
        isSelected =
            Dict.member mask.maskId model.selectedMasks

        currentLabel =
            Dict.get mask.maskId model.selectedMasks |> Maybe.withDefault 1
    in
    div
        [ class
            ("relative cursor-pointer border-2 rounded overflow-hidden "
                ++ (if isSelected then
                        "border-blue-500 ring-2 ring-blue-300"

                    else
                        "border-gray-200 hover:border-gray-400"
                   )
            )
        , onClick (ToggleMask mask.maskId)
        ]
        [ div
            [ class "w-full aspect-square relative"
            , style "background-color"
                (if isSelected then
                    maskColor (currentLabel + 1)

                 else
                    "#111827"
                )
            ]
            [ img
                [ src mask.url
                , class "w-full h-full object-contain"
                , style "mix-blend-mode"
                    (if isSelected then
                        "multiply"

                     else
                        "normal"
                    )
                ]
                []
            ]
        , div [ class "absolute bottom-0 left-0 right-0 bg-black bg-opacity-60 text-white text-xs p-1 flex justify-between items-center" ]
            [ span [] [ text (String.fromInt mask.maskId) ]
            , case mask.score of
                Just s ->
                    span [] [ text (String.fromFloat (toFloat (round (s * 100)) / 100)) ]

                Nothing ->
                    text ""
            ]
        , if isSelected then
            div [ class "absolute top-1 right-1" ]
                [ select
                    [ onInput (SetMaskLabel mask.maskId)
                    , stopPropagationOn "click" (D.succeed ( SetMaskLabel mask.maskId (indexToLetter currentLabel), True ))
                    , class "w-10 h-6 text-xs text-center border rounded bg-white"
                    ]
                    (List.map
                        (\i ->
                            option
                                [ value (indexToLetter i)
                                , selected (i == currentLabel)
                                ]
                                [ text (indexToLetter i) ]
                        )
                        (List.range 0 25)
                    )
                ]

          else
            text ""
        ]


fileDecoder : D.Decoder File
fileDecoder =
    D.at [ "target", "files" ] (D.index 0 File.decoder)


onEnter : Msg -> Html.Attribute Msg
onEnter msg =
    on "keydown"
        (D.field "key" D.string
            |> D.andThen
                (\key ->
                    if key == "Enter" then
                        D.succeed msg

                    else
                        D.fail "Not Enter"
                )
        )
